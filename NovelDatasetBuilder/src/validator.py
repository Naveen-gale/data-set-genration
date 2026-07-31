"""
validator.py
------------
Post-generation validation pass over the final JSONL dataset.

Responsibilities
----------------
1. Parse every line and verify it is valid JSON.
2. Check that every object has non-empty "instruction" and "response" fields.
3. Verify responses meet the minimum word count.
4. Detect and remove conversation-contaminated entries.
5. Strip invalid Unicode, extra whitespace, and bad escape sequences.
6. Repair repairable lines; drop irrecoverable ones.
7. Write a clean, validated version back to the output file (in-place).
"""

import json
import logging
import re
from pathlib import Path

from src.config import FINAL_DATASET, MIN_RESPONSE_WORDS
from src.utils import clean_whitespace, normalise_unicode, sanitise_json_string

logger = logging.getLogger(__name__)

# Conversation-style contamination prefixes (case-insensitive)
_CONVERSATION_PREFIXES = re.compile(
    r"^(?:student|teacher|professor|tutor|q|a|user|assistant)\s*:",
    re.IGNORECASE,
)


def _clean_string(value: str) -> str:
    """Normalise and clean a single string value from a JSON pair."""
    value = normalise_unicode(value)
    value = clean_whitespace(value)
    value = sanitise_json_string(value)
    return value


def _is_valid_pair(obj: dict) -> tuple[bool, str]:
    """
    Return (True, "") if *obj* passes all validation checks.
    Return (False, reason) otherwise.
    """
    instruction = obj.get("instruction", "")
    response = obj.get("response", "")

    if not isinstance(instruction, str) or not instruction.strip():
        return False, "empty instruction"
    if not isinstance(response, str) or not response.strip():
        return False, "empty response"

    # Word count check
    if len(response.split()) < MIN_RESPONSE_WORDS:
        return False, f"response too short ({len(response.split())} words)"

    # Conversation contamination
    if _CONVERSATION_PREFIXES.match(instruction.strip()):
        return False, "instruction has conversation prefix"
    if _CONVERSATION_PREFIXES.match(response.strip()):
        return False, "response has conversation prefix"

    # Instruction must not duplicate response (or vice versa)
    if instruction.strip().lower() == response.strip().lower():
        return False, "instruction equals response"

    return True, ""


def _try_parse_line(line: str) -> dict | None:
    """
    Attempt to parse a raw JSONL line.

    Returns the parsed dict or None if unrecoverable.
    """
    line = line.strip()
    if not line:
        return None

    # Direct parse
    try:
        obj = json.loads(line)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # Attempt to fix common escaping issues
    fixed = re.sub(r"\\(?![\"\\\/bfnrtu])", r"\\\\", line)
    try:
        obj = json.loads(fixed)
        if isinstance(obj, dict):
            logger.debug("Repaired line by fixing bad escapes.")
            return obj
    except json.JSONDecodeError:
        pass

    return None


class DatasetValidator:
    """
    Validates and repairs the dataset file in-place.

    Parameters
    ----------
    dataset_path:
        Path to the JSONL dataset file.
    min_response_words:
        Minimum word count for a valid response.
    """

    def __init__(
        self,
        dataset_path: Path = FINAL_DATASET,
        min_response_words: int = MIN_RESPONSE_WORDS,
    ) -> None:
        self.dataset_path = dataset_path
        self.min_response_words = min_response_words

    # ─── Public API ───────────────────────────────────────────────────────

    def validate_and_clean(self) -> tuple[int, int]:
        """
        Read the dataset, validate every line, clean strings, and overwrite
        the file with only valid entries.

        Returns
        -------
        tuple[int, int]
            (total_input_lines, valid_output_lines)
        """
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.dataset_path}")

        raw_lines = self.dataset_path.read_text(encoding="utf-8", errors="replace").splitlines()
        logger.info("Validator: reading %d lines from %s", len(raw_lines), self.dataset_path)

        valid_objects: list[dict] = []
        dropped = 0

        for i, line in enumerate(raw_lines, start=1):
            obj = _try_parse_line(line)
            if obj is None:
                logger.debug("Line %d: unparseable — dropped.", i)
                dropped += 1
                continue

            # Clean string fields
            if "instruction" in obj:
                obj["instruction"] = _clean_string(str(obj["instruction"]))
            if "response" in obj:
                obj["response"] = _clean_string(str(obj["response"]))

            ok, reason = _is_valid_pair(obj)
            if not ok:
                logger.debug("Line %d dropped: %s", i, reason)
                dropped += 1
                continue

            # Keep only the two required fields
            valid_objects.append(
                {"instruction": obj["instruction"], "response": obj["response"]}
            )

        logger.info(
            "Validation: %d valid / %d dropped from %d total lines.",
            len(valid_objects),
            dropped,
            len(raw_lines),
        )

        # Overwrite the file with clean data
        self.dataset_path.write_text(
            "\n".join(json.dumps(obj, ensure_ascii=False) for obj in valid_objects) + "\n",
            encoding="utf-8",
        )

        return len(raw_lines), len(valid_objects)


def validate_dataset(dataset_path: Path = FINAL_DATASET) -> tuple[int, int]:
    """
    Convenience wrapper around :class:`DatasetValidator`.

    Parameters
    ----------
    dataset_path:
        Path to the dataset file.

    Returns
    -------
    tuple[int, int]
        (total_input_lines, valid_output_lines)
    """
    validator = DatasetValidator(dataset_path)
    return validator.validate_and_clean()
