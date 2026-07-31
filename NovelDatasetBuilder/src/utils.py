"""
utils.py
--------
Shared utility helpers: logging setup, text normalisation, retry decorator,
and miscellaneous convenience functions used across the pipeline.
"""

import json
import logging
import re
import sys
import time
import unicodedata
from functools import wraps
from pathlib import Path
from typing import Any, Callable, TypeVar

from src.config import LOG_FILE, LOG_LEVEL, LLM_MAX_RETRIES, LLM_RETRY_DELAY

F = TypeVar("F", bound=Callable[..., Any])


# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────

def setup_logging() -> logging.Logger:
    """
    Configure the root logger to write to both stdout and a persistent log file.

    Returns
    -------
    logging.Logger
        The configured root logger instance.
    """
    log_level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(log_level)

    # Prevent duplicate handlers on repeated calls (e.g. during testing)
    if root.handlers:
        return root

    # Console handler — force UTF-8 on Windows (avoids cp1252 UnicodeEncodeError)
    import io
    utf8_stdout = io.TextIOWrapper(
        sys.stdout.buffer,
        encoding="utf-8",
        errors="replace",
        line_buffering=True,
    )
    ch = logging.StreamHandler(utf8_stdout)
    ch.setLevel(log_level)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    # File handler
    try:
        fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
        fh.setLevel(log_level)
        fh.setFormatter(fmt)
        root.addHandler(fh)
    except OSError:
        # If we cannot write the log file, fall back silently to console only
        pass

    return root


logger = setup_logging()


# ─────────────────────────────────────────────
# Retry decorator
# ─────────────────────────────────────────────

def retry(
    max_attempts: int = LLM_MAX_RETRIES,
    delay: float = LLM_RETRY_DELAY,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[F], F]:
    """
    Decorator that retries a function up to *max_attempts* times on specified
    exceptions, with exponential back-off starting from *delay* seconds.

    Parameters
    ----------
    max_attempts:
        Maximum number of total call attempts (including the first).
    delay:
        Base delay in seconds between attempts (doubles each retry).
    exceptions:
        Tuple of exception types that trigger a retry.
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            wait = delay
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    if attempt == max_attempts:
                        logger.error(
                            "Function '%s' failed after %d attempts: %s",
                            func.__name__,
                            max_attempts,
                            exc,
                        )
                        raise
                    logger.warning(
                        "Attempt %d/%d for '%s' failed: %s - retrying in %.1fs ...",
                        attempt,
                        max_attempts,
                        func.__name__,
                        exc,
                        wait,
                    )
                    time.sleep(wait)
                    wait = min(wait * 2, 60.0)  # cap at 60 s

        return wrapper  # type: ignore[return-value]

    return decorator


# ─────────────────────────────────────────────
# Text helpers
# ─────────────────────────────────────────────

def normalise_unicode(text: str) -> str:
    """Normalise Unicode to NFC form and remove control characters."""
    text = unicodedata.normalize("NFC", text)
    # Remove non-printable control characters except newlines/tabs
    text = re.sub(r"[^\S\n\t]", " ", text)  # collapse weird whitespace to space
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    return text


def clean_whitespace(text: str) -> str:
    """Collapse runs of spaces/tabs and strip each line."""
    lines = text.splitlines()
    cleaned: list[str] = []
    for line in lines:
        line = re.sub(r"[ \t]+", " ", line).strip()
        cleaned.append(line)
    # Collapse 3+ consecutive blank lines into 2
    text = "\n".join(cleaned)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def count_words(text: str) -> int:
    """Return the number of whitespace-separated tokens in *text*."""
    return len(text.split())


def sanitise_json_string(value: str) -> str:
    """
    Ensure a string is safe for embedding inside JSON.

    Replaces invalid escape sequences and removes stray control characters.
    """
    # Replace backslashes that are NOT part of a valid JSON escape
    value = re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", value)
    # Remove residual control characters (0x00–0x1f except \n, \r, \t)
    value = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", value)
    return value


def safe_json_dumps(obj: dict[str, Any]) -> str:
    """Serialise *obj* to a compact JSON string, ensuring ASCII safety."""
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def load_text_file(path: Path) -> str:
    """Read *path* as UTF-8 text; raise FileNotFoundError with a clear message."""
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return path.read_text(encoding="utf-8")
