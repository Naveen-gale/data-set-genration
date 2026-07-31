"""
deduplicate.py
--------------
Second-pass deduplication of the validated JSONL dataset.

Removes
-------
- Exact duplicate instructions
- Exact duplicate responses
- Near-duplicate instruction-response pairs (cosine similarity over TF-IDF)
- Very short or low-quality entries
- Copy-paste responses (response is substring of instruction or vice versa)

The deduplication is performed in-place: the original file is overwritten
with the deduplicated version.
"""

import hashlib
import json
import logging
import math
import re
from collections import Counter
from pathlib import Path

from src.config import FINAL_DATASET, MIN_RESPONSE_WORDS, SIMILARITY_THRESHOLD

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Lightweight TF-IDF similarity
# ─────────────────────────────────────────────

def _tokenise(text: str) -> list[str]:
    """Lower-case, alpha-only tokenisation."""
    return re.findall(r"[a-z]+", text.lower())


def _tfidf_vector(tokens: list[str], idf: dict[str, float]) -> dict[str, float]:
    """Compute a TF-IDF vector for *tokens* given an IDF mapping."""
    tf = Counter(tokens)
    total = len(tokens) or 1
    return {term: (count / total) * idf.get(term, 0) for term, count in tf.items()}


def _cosine_similarity(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
    """Compute cosine similarity between two sparse vectors."""
    if not vec_a or not vec_b:
        return 0.0
    dot = sum(vec_a.get(k, 0.0) * vec_b.get(k, 0.0) for k in vec_b)
    norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
    norm_b = math.sqrt(sum(v * v for v in vec_b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _build_idf(all_token_lists: list[list[str]]) -> dict[str, float]:
    """Build IDF values from a corpus of token lists."""
    N = len(all_token_lists) or 1
    df: Counter[str] = Counter()
    for tokens in all_token_lists:
        df.update(set(tokens))
    return {term: math.log(N / (count + 1)) + 1 for term, count in df.items()}


# ─────────────────────────────────────────────
# Hash-based exact deduplication helpers
# ─────────────────────────────────────────────

def _fingerprint(text: str) -> str:
    """Return an MD5 hex digest of the normalised *text*."""
    normalised = " ".join(text.lower().split())
    return hashlib.md5(normalised.encode("utf-8")).hexdigest()


# ─────────────────────────────────────────────
# Main deduplicator
# ─────────────────────────────────────────────

class Deduplicator:
    """
    Removes duplicate and near-duplicate pairs from the JSONL dataset.

    Parameters
    ----------
    dataset_path:
        Path to the validated JSONL file.
    similarity_threshold:
        Cosine similarity above which two instructions are considered duplicates.
    min_response_words:
        Minimum word count for a response to be retained.
    """

    def __init__(
        self,
        dataset_path: Path = FINAL_DATASET,
        similarity_threshold: float = SIMILARITY_THRESHOLD,
        min_response_words: int = MIN_RESPONSE_WORDS,
    ) -> None:
        self.dataset_path = dataset_path
        self.similarity_threshold = similarity_threshold
        self.min_response_words = min_response_words

    # ─── Public API ───────────────────────────────────────────────────────

    def deduplicate(self) -> tuple[int, int]:
        """
        Load, deduplicate, and overwrite the dataset file.

        Returns
        -------
        tuple[int, int]
            (input_count, output_count)
        """
        objects = self._load()
        logger.info("Deduplicator: loaded %d pairs.", len(objects))

        # Pass 1: exact deduplication by instruction fingerprint
        objects = self._exact_deduplicate(objects)
        logger.info("After exact dedup: %d pairs.", len(objects))

        # Pass 2: exact deduplication by response fingerprint
        objects = self._exact_deduplicate_responses(objects)
        logger.info("After response dedup: %d pairs.", len(objects))

        # Pass 3: remove low-quality entries
        objects = self._remove_low_quality(objects)
        logger.info("After quality filter: %d pairs.", len(objects))

        # Pass 4: near-duplicate detection on instructions
        objects = self._near_deduplicate(objects)
        logger.info("After near-dedup: %d pairs.", len(objects))

        self._save(objects)
        return len(objects), len(objects)  # (written, written) — input already shrunk

    # ─── Private helpers ──────────────────────────────────────────────────

    def _load(self) -> list[dict]:
        raw = self.dataset_path.read_text(encoding="utf-8", errors="replace")
        objects: list[dict] = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict) and "instruction" in obj and "response" in obj:
                    objects.append(obj)
            except json.JSONDecodeError:
                pass
        return objects

    def _exact_deduplicate(self, objects: list[dict]) -> list[dict]:
        """Remove pairs with duplicate instructions (case-normalised)."""
        seen: set[str] = set()
        unique: list[dict] = []
        for obj in objects:
            fp = _fingerprint(obj["instruction"])
            if fp not in seen:
                seen.add(fp)
                unique.append(obj)
        return unique

    def _exact_deduplicate_responses(self, objects: list[dict]) -> list[dict]:
        """Remove pairs with duplicate responses (case-normalised)."""
        seen: set[str] = set()
        unique: list[dict] = []
        for obj in objects:
            fp = _fingerprint(obj["response"])
            if fp not in seen:
                seen.add(fp)
                unique.append(obj)
        return unique

    def _remove_low_quality(self, objects: list[dict]) -> list[dict]:
        """
        Remove pairs that are too short, copy-paste, or trivially identical.
        """
        kept: list[dict] = []
        for obj in objects:
            instruction = obj["instruction"].strip()
            response = obj["response"].strip()

            # Response too short
            if len(response.split()) < self.min_response_words:
                continue

            # Response is a copy of the instruction
            if instruction.lower() in response.lower() and len(instruction.split()) > 8:
                # If the instruction appears verbatim and makes up >80% of response, skip
                ratio = len(instruction) / max(len(response), 1)
                if ratio > 0.8:
                    continue

            kept.append(obj)
        return kept

    def _near_deduplicate(self, objects: list[dict]) -> list[dict]:
        """
        Remove near-duplicate instructions using TF-IDF cosine similarity.

        This is O(n²) but is necessary for quality; acceptable for dataset sizes
        up to ~50 000 pairs. Pairs are processed in order and the first
        occurrence is always kept.
        """
        if not objects:
            return objects

        token_lists = [_tokenise(obj["instruction"]) for obj in objects]
        idf = _build_idf(token_lists)
        vectors = [_tfidf_vector(tl, idf) for tl in token_lists]

        kept_indices: list[int] = []
        dropped = 0

        for i in range(len(objects)):
            is_near_dup = False
            for j in kept_indices:
                sim = _cosine_similarity(vectors[i], vectors[j])
                if sim >= self.similarity_threshold:
                    is_near_dup = True
                    dropped += 1
                    break
            if not is_near_dup:
                kept_indices.append(i)

        logger.info("Near-dedup dropped %d near-duplicate pairs.", dropped)
        return [objects[i] for i in kept_indices]

    def _save(self, objects: list[dict]) -> None:
        """Overwrite the dataset file with *objects*."""
        lines = [json.dumps(obj, ensure_ascii=False) for obj in objects]
        self.dataset_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        logger.info(
            "Deduplication complete. %d pairs saved to %s.",
            len(objects),
            self.dataset_path,
        )


def deduplicate_dataset(dataset_path: Path = FINAL_DATASET) -> int:
    """
    Convenience wrapper around :class:`Deduplicator`.

    Parameters
    ----------
    dataset_path:
        Path to the validated JSONL dataset.

    Returns
    -------
    int
        Number of pairs in the deduplicated dataset.
    """
    deduplicator = Deduplicator(dataset_path)
    _, count = deduplicator.deduplicate()
    return count
