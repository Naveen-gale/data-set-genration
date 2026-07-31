"""
main.py
-------
Entry point for the NovelDatasetBuilder pipeline.

Pipeline stages
---------------
1. Validate configuration (API key, paths).
2. Extract text from the input PDF.
3. Detect structural sections (chapters, scenes, acts …).
4. Build intelligent text chunks.
5. Generate instruction-response pairs via the LLM (streamed to output file).
6. Validate the raw dataset (repair / drop broken entries).
7. Deduplicate (exact + near-duplicate).
8. Print a final summary.

Usage
-----
    python -m src.main              # fresh run (deletes previous output)
    python -m src.main --resume     # resume interrupted run (keeps existing output)
or
    python src/main.py
    python src/main.py --resume
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# ── Initialise logging before any other module imports ──────────────────────
from src.utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

# ── Project imports ──────────────────────────────────────────────────────────
from src.config import (
    FINAL_DATASET,
    INPUT_PDF,
    validate_config,
)
from src.extract_pdf import extract_pdf
from src.detect_sections import detect_sections
from src.chunk_builder import build_chunks
from src.llm_generator import generate_dataset
from src.validator import validate_dataset
from src.deduplicate import deduplicate_dataset


def _banner(message: str) -> None:
    """Print a visually distinct pipeline stage banner."""
    sep = "═" * 70
    logger.info("\n%s\n  %s\n%s", sep, message, sep)


def _check_pdf(pdf_path: Path) -> None:
    """Ensure the input PDF exists and is non-empty before starting."""
    if not pdf_path.exists():
        logger.error(
            "Input PDF not found: %s\n"
            "Please copy your novel PDF to the 'input/' directory as 'novel.pdf'.",
            pdf_path,
        )
        sys.exit(1)
    if pdf_path.stat().st_size < 1024:
        logger.error("Input PDF appears to be empty or corrupt: %s", pdf_path)
        sys.exit(1)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NovelDatasetBuilder — Generate fine-tuning datasets from a novel PDF."
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help=(
            "Resume an interrupted run. Keeps existing output and skips "
            "chunks that were already processed (tracked via checkpoint file). "
            "Without this flag, any existing output is deleted for a fresh start."
        ),
    )
    return parser.parse_args()


def run_pipeline(resume: bool = False) -> None:
    """Execute the full NovelDatasetBuilder pipeline."""
    start_time = time.time()

    # ── Stage 0: Configuration validation ───────────────────────────────────
    _banner("Stage 0 │ Validating configuration")
    try:
        validate_config()
    except EnvironmentError as exc:
        logger.error("Configuration error: %s", exc)
        sys.exit(1)

    _check_pdf(INPUT_PDF)

    # Handle existing dataset based on run mode
    if FINAL_DATASET.exists():
        if resume:
            existing_lines = sum(
                1 for line in FINAL_DATASET.read_text(encoding="utf-8", errors="replace").splitlines()
                if line.strip()
            )
            logger.info(
                "RESUME MODE: Keeping existing dataset (%d pairs). "
                "Will skip already-processed chunks.",
                existing_lines,
            )
        else:
            logger.warning(
                "Fresh run: removing existing dataset at %s. "
                "Use --resume to continue an interrupted run.",
                FINAL_DATASET,
            )
            FINAL_DATASET.unlink()
    elif resume:
        logger.info("RESUME MODE: No existing output found — starting fresh.")

    # ── Stage 1: PDF extraction ──────────────────────────────────────────────
    _banner("Stage 1 │ Extracting PDF text")
    novel_text = extract_pdf(INPUT_PDF)
    word_count = len(novel_text.split())
    logger.info("Extracted %d words from the novel.", word_count)

    if word_count < 500:
        logger.error(
            "Extracted text is too short (%d words). "
            "The PDF may be scanned/image-based (not text-selectable). "
            "Consider running OCR on the PDF first.",
            word_count,
        )
        sys.exit(1)

    # ── Stage 2: Section detection ───────────────────────────────────────────
    _banner("Stage 2 │ Detecting novel sections")
    sections = detect_sections(novel_text)
    logger.info("Detected %d sections.", len(sections))

    if not sections:
        logger.error("No sections detected. Check the extracted text.")
        sys.exit(1)

    # ── Stage 3: Chunk building ──────────────────────────────────────────────
    _banner("Stage 3 │ Building text chunks")
    chunks = build_chunks(sections)
    logger.info(
        "Built %d chunks. Average size: %d words.",
        len(chunks),
        word_count // max(len(chunks), 1),
    )

    if not chunks:
        logger.error("No chunks produced. Check the section detection output.")
        sys.exit(1)

    # ── Stage 4: LLM dataset generation ─────────────────────────────────────
    _banner("Stage 4 │ Generating instruction-response pairs via LLM")
    total_generated = generate_dataset(chunks, resume=resume)
    logger.info("Raw pairs generated/accumulated: %d", total_generated)

    if total_generated == 0:
        logger.error(
            "LLM generation produced zero valid pairs. "
            "Check your API key, model name, and network connection."
        )
        sys.exit(1)

    # ── Stage 5: Validation ──────────────────────────────────────────────────
    _banner("Stage 5 │ Validating dataset")
    total_in, total_valid = validate_dataset(FINAL_DATASET)
    logger.info("Validation: %d in → %d valid.", total_in, total_valid)

    # ── Stage 6: Deduplication ───────────────────────────────────────────────
    _banner("Stage 6 │ Deduplicating dataset")
    final_count = deduplicate_dataset(FINAL_DATASET)
    logger.info("Deduplicated dataset size: %d pairs.", final_count)

    # ── Final summary ────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    _banner("Pipeline Complete")
    logger.info(
        "\n"
        "  ✓ Input PDF          : %s\n"
        "  ✓ Sections detected  : %d\n"
        "  ✓ Chunks processed   : %d\n"
        "  ✓ Pairs generated    : %d\n"
        "  ✓ Pairs after valid  : %d\n"
        "  ✓ Final dataset size : %d pairs\n"
        "  ✓ Output file        : %s\n"
        "  ✓ Time elapsed       : %.1f seconds",
        INPUT_PDF,
        len(sections),
        len(chunks),
        total_generated,
        total_valid,
        final_count,
        FINAL_DATASET,
        elapsed,
    )


if __name__ == "__main__":
    args = _parse_args()
    run_pipeline(resume=args.resume)
