"""
config.py
---------
Central configuration for the NovelDatasetBuilder pipeline.
Supports both OpenAI and Groq (via OpenAI-compatible SDK).
All parameters are loaded from environment variables with sensible defaults.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from the .env file in project root
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")


# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
PROJECT_ROOT: Path = Path(__file__).parent.parent
INPUT_DIR: Path = PROJECT_ROOT / "input"
OUTPUT_DIR: Path = PROJECT_ROOT / "output"
PROMPTS_DIR: Path = PROJECT_ROOT / "prompts"

INPUT_PDF: Path = INPUT_DIR / "novel.pdf"
FINAL_DATASET: Path = OUTPUT_DIR / "final_dataset.jsonl"

GENERATION_PROMPT_FILE: Path = PROMPTS_DIR / "generation_prompt.txt"


# ─────────────────────────────────────────────
# API Provider Detection
# ─────────────────────────────────────────────
# The pipeline auto-detects which provider to use based on which key is set.
# Groq takes priority when GROQ_API_KEY is present.

GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

# Set whichever key is available as the active key
ACTIVE_API_KEY: str = GROQ_API_KEY or OPENAI_API_KEY

# Groq uses the OpenAI SDK with a custom base URL
GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"
USE_GROQ: bool = bool(GROQ_API_KEY)


# ─────────────────────────────────────────────
# LLM Settings
# ─────────────────────────────────────────────
# Default model differs by provider:
#   Groq  → llama-3.3-70b-versatile  (free, high quality)
#   OpenAI→ gpt-4o
_default_model = "llama-3.3-70b-versatile" if USE_GROQ else "gpt-4o"
LLM_MODEL: str = os.getenv("LLM_MODEL", _default_model)

LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.75"))

# Keep max_tokens within Groq free-tier output limits per call
LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "4096"))

# Retry settings
LLM_MAX_RETRIES: int = int(os.getenv("LLM_MAX_RETRIES", "6"))
LLM_RETRY_DELAY: float = float(os.getenv("LLM_RETRY_DELAY", "5.0"))


# ─────────────────────────────────────────────
# Rate-Limit Protection (Groq Free Tier)
# ─────────────────────────────────────────────
# Groq free tier: 30 req/min, ~6 000 tokens/min output.
# We add a mandatory inter-request delay to avoid 429 errors.
# Default 12s → max ~5 calls/min → ~5*2400 = 12 000 tokens/min (with headroom).

GROQ_INTER_REQUEST_DELAY: float = float(
    os.getenv("GROQ_INTER_REQUEST_DELAY", "12.0" if USE_GROQ else "0.0")
)

# How many pairs to request per individual LLM call.
# Groq: 20 pairs × ~120 tokens/pair = ~2 400 output tokens → safe per call.
PAIRS_PER_BATCH: int = int(os.getenv("PAIRS_PER_BATCH", "20" if USE_GROQ else "40"))


# ─────────────────────────────────────────────
# Dataset Generation
# ─────────────────────────────────────────────
# Total target pairs per chunk.
# Multiple batched API calls will be made per chunk if needed.
PAIRS_PER_CHUNK: int = int(os.getenv("PAIRS_PER_CHUNK", "40"))


# ─────────────────────────────────────────────
# Chunking
# ─────────────────────────────────────────────
CHUNK_MIN_WORDS: int = int(os.getenv("CHUNK_MIN_WORDS", "800"))
CHUNK_MAX_WORDS: int = int(os.getenv("CHUNK_MAX_WORDS", "1800"))
CHUNK_OVERLAP_WORDS: int = int(os.getenv("CHUNK_OVERLAP_WORDS", "100"))


# ─────────────────────────────────────────────
# Validation / Deduplication
# ─────────────────────────────────────────────
MIN_RESPONSE_WORDS: int = int(os.getenv("MIN_RESPONSE_WORDS", "20"))
SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.85"))


# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE: Path = PROJECT_ROOT / "pipeline.log"


def validate_config() -> None:
    """Raise early if any critical configuration is missing."""
    if not ACTIVE_API_KEY:
        raise EnvironmentError(
            "No API key found.\n"
            "  For Groq:   set GROQ_API_KEY=gsk_... in your .env file\n"
            "  For OpenAI: set OPENAI_API_KEY=sk-... in your .env file"
        )
    if USE_GROQ:
        import logging
        logging.getLogger(__name__).info(
            "Provider: GROQ | Model: %s | Batch size: %d pairs | "
            "Inter-request delay: %.1fs",
            LLM_MODEL,
            PAIRS_PER_BATCH,
            GROQ_INTER_REQUEST_DELAY,
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
