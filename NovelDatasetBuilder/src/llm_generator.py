"""
llm_generator.py
----------------
Calls the configured LLM (Groq or OpenAI) for each chunk and writes
validated {"instruction": "...", "response": "..."} pairs to the output file.

Groq free-tier design
---------------------
- Uses OpenAI SDK with Groq's base URL (fully compatible).
- SDK built-in retries are DISABLED (max_retries=0).  All retry/delay logic
  is handled manually so we have full control over request pacing.
- Each API call requests PAIRS_PER_BATCH pairs (default 10).
- A mandatory inter-request delay (GROQ_INTER_REQUEST_DELAY) is inserted
  before every API call to stay well within Groq's 30 RPM free-tier limit.
- On RateLimitError (HTTP 429) the generator reads the Retry-After header
  (or falls back to _RATE_LIMIT_COOLDOWN) then retries.
- Pairs are appended to the output file immediately after each batch so no
  data is lost on interruption.
- Resume support: already-processed chunk indices are tracked in a
  checkpoint file so the pipeline can be restarted without re-doing work.
"""

import json
import logging
import re
import time
from math import ceil
from pathlib import Path
from typing import Any

from openai import OpenAI, RateLimitError, APIStatusError
from tqdm import tqdm

from src.chunk_builder import Chunk
from src.config import (
    ACTIVE_API_KEY,
    FINAL_DATASET,
    GENERATION_PROMPT_FILE,
    GROQ_BASE_URL,
    GROQ_INTER_REQUEST_DELAY,
    LLM_MAX_RETRIES,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_RETRY_DELAY,
    LLM_TEMPERATURE,
    OUTPUT_DIR,
    PAIRS_PER_BATCH,
    PAIRS_PER_CHUNK,
    USE_GROQ,
)
from src.utils import load_text_file, safe_json_dumps

logger = logging.getLogger(__name__)

# Type alias
Pair = dict[str, str]

# Seconds to wait after a 429 before retrying (fallback if no Retry-After header)
_RATE_LIMIT_COOLDOWN: float = 65.0

# Maximum seconds we are willing to wait on a single 429 retry.
# If Groq asks for longer (e.g. daily quota reset ~32 min), we skip and save progress.
_MAX_WAIT_SECONDS: float = 120.0

# Checkpoint file tracks completed chunk indices for resume support
_CHECKPOINT_FILE: Path = OUTPUT_DIR / ".generation_checkpoint.json"


# ─────────────────────────────────────────────
# JSON extraction / repair helpers
# ─────────────────────────────────────────────

def _extract_json_array(raw: str) -> str:
    """Extract the outermost JSON array from the LLM response string."""
    raw = re.sub(r"```(?:json)?", "", raw, flags=re.IGNORECASE).strip()
    raw = raw.replace("```", "").strip()

    start = raw.find("[")
    if start == -1:
        return raw

    depth = 0
    for i, ch in enumerate(raw[start:], start):
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return raw[start : i + 1]
    return raw[start:]  # unclosed array -- best effort


def _repair_json(raw: str) -> list[Pair]:
    """
    Three-pass JSON recovery:
      1. Direct parse of extracted array.
      2. Strip trailing commas and re-parse.
      3. Extract individual {...} objects with regex.
    """
    array_str = _extract_json_array(raw)

    # Pass 1
    try:
        data = json.loads(array_str)
        if isinstance(data, list):
            return [d for d in data if isinstance(d, dict)]
    except json.JSONDecodeError:
        pass

    # Pass 2
    fixed = re.sub(r",\s*([}\]])", r"\1", array_str)
    try:
        data = json.loads(fixed)
        if isinstance(data, list):
            logger.debug("JSON repaired by removing trailing commas.")
            return [d for d in data if isinstance(d, dict)]
    except json.JSONDecodeError:
        pass

    # Pass 3
    objects: list[Pair] = []
    for m in re.finditer(r"\{[^{}]+\}", array_str, re.DOTALL):
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                objects.append(obj)
        except json.JSONDecodeError:
            continue

    if objects:
        logger.warning("Partial JSON repair: recovered %d objects via regex.", len(objects))
    return objects


# ─────────────────────────────────────────────
# Pair validation
# ─────────────────────────────────────────────

_BAD_PREFIXES = re.compile(
    r"^(?:student|teacher|professor|tutor|q|a|user|assistant)\s*:",
    re.IGNORECASE,
)


def _validate_pair(pair: Any) -> tuple[bool, str]:
    """Return (True, '') if the pair is usable, else (False, reason)."""
    if not isinstance(pair, dict):
        return False, "not a dict"
    instruction: str = pair.get("instruction", "")
    response: str = pair.get("response", "")
    if not isinstance(instruction, str) or not instruction.strip():
        return False, "empty instruction"
    if not isinstance(response, str) or not response.strip():
        return False, "empty response"
    if len(response.split()) < 20:
        return False, "response too short (%d words)" % len(response.split())
    if _BAD_PREFIXES.match(instruction.strip()):
        return False, "instruction has conversation prefix"
    return True, ""


# ─────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────

def _load_checkpoint() -> set[int]:
    """Load the set of already-completed chunk indices from disk."""
    if _CHECKPOINT_FILE.exists():
        try:
            data = json.loads(_CHECKPOINT_FILE.read_text(encoding="utf-8"))
            completed = set(data.get("completed_chunks", []))
            if completed:
                logger.info(
                    "Resume: found checkpoint — %d chunks already done: %s",
                    len(completed), sorted(completed),
                )
            return completed
        except Exception as exc:
            logger.warning("Could not read checkpoint file: %s — starting fresh.", exc)
    return set()


def _save_checkpoint(completed_chunks: set[int]) -> None:
    """Persist the set of completed chunk indices to disk."""
    try:
        _CHECKPOINT_FILE.write_text(
            json.dumps({"completed_chunks": sorted(completed_chunks)}),
            encoding="utf-8",
        )
    except Exception as exc:
        logger.warning("Could not write checkpoint: %s", exc)


def _clear_checkpoint() -> None:
    """Remove the checkpoint file after a successful run."""
    try:
        if _CHECKPOINT_FILE.exists():
            _CHECKPOINT_FILE.unlink()
    except Exception:
        pass


# ─────────────────────────────────────────────
# LLM client factory
# ─────────────────────────────────────────────

def _build_client() -> OpenAI:
    """
    Build an OpenAI-compatible client.

    For Groq: points at Groq's base URL with max_retries=0 so that ONLY
    our manual delay logic controls request pacing (no SDK auto-retry
    firing extra calls that trigger 429s).
    """
    if USE_GROQ:
        logger.info(
            "Provider: GROQ | Model: %s | Delay: %.0fs | Batch: %d pairs",
            LLM_MODEL, GROQ_INTER_REQUEST_DELAY, PAIRS_PER_BATCH,
        )
        return OpenAI(
            api_key=ACTIVE_API_KEY,
            base_url=GROQ_BASE_URL,
            max_retries=0,      # disable SDK auto-retry — we retry manually
            timeout=120.0,      # generous timeout for slow responses
        )
    logger.info("Provider: OpenAI | Model: %s", LLM_MODEL)
    return OpenAI(api_key=ACTIVE_API_KEY, max_retries=2, timeout=60.0)


# ─────────────────────────────────────────────
# Main generator class
# ─────────────────────────────────────────────

class LLMGenerator:
    """
    Generates instruction-response pairs for every text chunk.

    Groq free-tier pacing
    ---------------------
    - PAIRS_PER_BATCH = 10  => ~1200 output tokens/call
    - GROQ_INTER_REQUEST_DELAY = 35s => 1.7 calls/min max
    - On 429: read Retry-After header, sleep that long, then retry
    - Resume: skip already-completed chunks using checkpoint file
    """

    def __init__(
        self,
        output_path: Path = FINAL_DATASET,
        prompt_template: str | None = None,
        resume: bool = True,
    ) -> None:
        self.output_path = output_path
        self.client = _build_client()
        self.prompt_template = prompt_template or load_text_file(GENERATION_PROMPT_FILE)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._last_call_time: float = 0.0
        self._resume = resume

    # ─── Public ───────────────────────────────────────────────────────────

    def process_chunks(self, chunks: list[Chunk]) -> int:
        """Process all chunks and return total pairs written."""
        batches_per_chunk = ceil(PAIRS_PER_CHUNK / PAIRS_PER_BATCH)
        total_calls = len(chunks) * batches_per_chunk
        est_mins = (total_calls * GROQ_INTER_REQUEST_DELAY) / 60

        logger.info(
            "Plan: %d chunks x %d batches x %d pairs = ~%d pairs | "
            "Delay: %.0fs/call | Est: ~%.0f min",
            len(chunks), batches_per_chunk, PAIRS_PER_BATCH,
            len(chunks) * PAIRS_PER_CHUNK,
            GROQ_INTER_REQUEST_DELAY, est_mins,
        )

        # Load resume checkpoint
        completed_chunks: set[int] = _load_checkpoint() if self._resume else set()

        # Count how many pairs are already written (for accurate progress display)
        total_written = self._count_existing_pairs()
        if completed_chunks:
            logger.info(
                "Resuming: %d pairs already in output file, skipping %d completed chunks.",
                total_written, len(completed_chunks),
            )

        pending_calls = sum(
            batches_per_chunk for chunk in chunks
            if chunk.chunk_index not in completed_chunks
        )

        with tqdm(total=total_calls, desc="Generating", unit="call",
                  dynamic_ncols=True, initial=total_calls - pending_calls) as pbar:
            for chunk in chunks:
                if chunk.chunk_index in completed_chunks:
                    pbar.update(batches_per_chunk)
                    continue

                chunk_pairs: list[Pair] = []
                seen: set[str] = set()

                for batch_idx in range(batches_per_chunk):
                    remaining = PAIRS_PER_CHUNK - len(chunk_pairs)
                    batch_size = min(PAIRS_PER_BATCH, remaining)
                    if batch_size <= 0:
                        pbar.update(1)
                        continue

                    self._wait_for_rate_limit()

                    pairs = self._generate_batch(chunk, batch_size, batch_idx, seen)

                    for p in pairs:
                        key = p["instruction"].strip().lower()
                        if key not in seen:
                            seen.add(key)
                            chunk_pairs.append(p)

                    pbar.set_postfix({
                        "chunk": chunk.chunk_index,
                        "pairs": total_written + len(chunk_pairs),
                    })
                    pbar.update(1)

                written = self._write_pairs(chunk_pairs)
                total_written += written

                # Mark chunk as complete and persist checkpoint
                completed_chunks.add(chunk.chunk_index)
                _save_checkpoint(completed_chunks)

        logger.info("Generation complete. Total pairs: %d", total_written)
        _clear_checkpoint()
        return total_written

    # ─── Private ──────────────────────────────────────────────────────────

    def _count_existing_pairs(self) -> int:
        """Count lines in the output file (to track existing progress)."""
        if not self.output_path.exists():
            return 0
        try:
            with self.output_path.open("r", encoding="utf-8") as fh:
                return sum(1 for line in fh if line.strip())
        except Exception:
            return 0

    def _wait_for_rate_limit(self) -> None:
        """Sleep only the remaining portion of the inter-request delay."""
        if GROQ_INTER_REQUEST_DELAY <= 0:
            return
        elapsed = time.monotonic() - self._last_call_time
        gap = GROQ_INTER_REQUEST_DELAY - elapsed
        if gap > 0:
            logger.debug("Rate-limit pause: %.1fs", gap)
            time.sleep(gap)

    def _build_prompt(self, chunk: Chunk, num_pairs: int,
                      existing: set[str]) -> str:
        """Format prompt, injecting already-seen instructions to avoid dupes."""
        avoid = ""
        if existing:
            sample = list(existing)[:12]
            avoid = (
                "\n\nALREADY GENERATED (do NOT repeat or rephrase):\n"
                + "\n".join("- " + s[:80] for s in sample)
            )
        return self.prompt_template.format(
            num_pairs=num_pairs,
            chunk_text=chunk.text,
            source_section=chunk.source_section,
            section_type=chunk.section_type,
        ) + avoid

    def _call_api(self, prompt: str) -> str:
        """Single API call. Records time. Raises exceptions for caller to handle."""
        self._last_call_time = time.monotonic()
        resp = self.client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert academic dataset engineer. "
                        "Output ONLY a valid JSON array. No markdown. No explanation."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
        )
        content = (resp.choices[0].message.content or "").strip()
        if not content:
            raise RuntimeError("LLM returned empty content.")
        return content

    def _generate_batch(self, chunk: Chunk, num_pairs: int,
                        batch_idx: int, existing: set[str]) -> list[Pair]:
        """Call LLM with retry logic. Returns validated pairs."""
        prompt = self._build_prompt(chunk, num_pairs, existing)

        for attempt in range(1, LLM_MAX_RETRIES + 1):
            try:
                raw = self._call_api(prompt)
                break  # success — exit retry loop

            except RateLimitError as exc:
                # Try to read Retry-After from response headers
                retry_after = _RATE_LIMIT_COOLDOWN
                try:
                    if hasattr(exc, "response") and exc.response is not None:
                        ra = exc.response.headers.get("retry-after")
                        if ra:
                            retry_after = float(ra) + 2.0  # small buffer
                except Exception:
                    pass

                # If Groq asks us to wait longer than our max (daily quota hit),
                # skip this batch and save progress rather than hanging.
                if retry_after > _MAX_WAIT_SECONDS:
                    logger.error(
                        "429: Groq asks to wait %.0fs (%.1f min) — daily token quota likely hit.\n"
                        "  Progress is SAVED via checkpoint. Re-run with --resume after %.0f minutes.",
                        retry_after, retry_after / 60, retry_after / 60,
                    )
                    return []  # skip batch, outer loop will checkpoint and exit

                logger.warning(
                    "429 Rate limit (attempt %d/%d). Waiting %.0fs ...",
                    attempt, LLM_MAX_RETRIES, retry_after,
                )
                time.sleep(retry_after)
                self._last_call_time = time.monotonic()

                if attempt == LLM_MAX_RETRIES:
                    logger.error(
                        "Chunk %d batch %d: max retries reached after 429, skipping.",
                        chunk.chunk_index, batch_idx,
                    )
                    return []

            except APIStatusError as exc:
                wait = min(LLM_RETRY_DELAY * (2 ** (attempt - 1)), 120.0)
                logger.warning(
                    "API %d error (attempt %d/%d). Waiting %.0fs ...",
                    exc.status_code, attempt, LLM_MAX_RETRIES, wait,
                )
                time.sleep(wait)
                if attempt == LLM_MAX_RETRIES:
                    return []

            except Exception as exc:
                wait = min(LLM_RETRY_DELAY * (2 ** (attempt - 1)), 120.0)
                logger.warning(
                    "Error (attempt %d/%d): %s. Waiting %.0fs ...",
                    attempt, LLM_MAX_RETRIES, exc, wait,
                )
                time.sleep(wait)
                if attempt == LLM_MAX_RETRIES:
                    return []
        else:
            return []

        # Validate
        parsed = _repair_json(raw)
        valid: list[Pair] = []
        for pair in parsed:
            ok, reason = _validate_pair(pair)
            if ok:
                valid.append({
                    "instruction": pair["instruction"].strip(),
                    "response": pair["response"].strip(),
                })
            else:
                logger.debug("Dropped pair: %s", reason)

        logger.info(
            "Chunk %d | Batch %d -> %d/%d pairs valid",
            chunk.chunk_index, batch_idx, len(valid), len(parsed),
        )
        return valid

    def _write_pairs(self, pairs: list[Pair]) -> int:
        """Append pairs to output JSONL immediately."""
        written = 0
        with self.output_path.open("a", encoding="utf-8") as fh:
            for pair in pairs:
                try:
                    fh.write(safe_json_dumps(pair) + "\n")
                    written += 1
                except (TypeError, ValueError) as exc:
                    logger.warning("Serialise error: %s", exc)
        return written


def generate_dataset(chunks: list[Chunk], resume: bool = True) -> int:
    """Convenience wrapper: generate dataset from chunks with optional resume."""
    return LLMGenerator(resume=resume).process_chunks(chunks)
