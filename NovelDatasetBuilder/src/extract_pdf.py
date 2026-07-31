"""
extract_pdf.py
--------------
Reads the novel PDF, removes noise (page numbers, headers, footers, OCR
artefacts) and returns clean plain text ready for downstream processing.
"""

import logging
import re
from pathlib import Path

import pdfplumber

from src.utils import clean_whitespace, normalise_unicode

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Heuristics for noise detection
# ─────────────────────────────────────────────

# Lines that are almost certainly page numbers (standalone integer, Roman numeral, etc.)
_PAGE_NUMBER_RE = re.compile(
    r"^(?:page\s*)?\d{1,4}$|^[ivxlcdmIVXLCDM]{1,8}$", re.IGNORECASE
)

# Very short lines that likely represent running headers/footers (≤ 6 words)
_SHORT_LINE_WORD_THRESHOLD = 6

# Patterns that indicate a pure OCR artefact or garbage line
_GARBAGE_RE = re.compile(r"^[\W_]{4,}$")  # all non-word chars


def _is_noise_line(line: str) -> bool:
    """
    Return True if *line* is a noise line that should be discarded.

    Criteria
    --------
    - Matches a page-number pattern.
    - Is a short line (≤ 6 words) that appears more than twice on the page
      (detected by the caller aggregating repetitions).
    - Consists entirely of non-word characters (garbage OCR output).
    """
    stripped = line.strip()
    if not stripped:
        return True
    if _PAGE_NUMBER_RE.match(stripped):
        return True
    if _GARBAGE_RE.match(stripped):
        return True
    return False


def _remove_headers_footers(pages_text: list[str]) -> list[str]:
    """
    Detect and strip recurring header/footer lines across pages.

    A line is considered a header or footer if it appears (verbatim, stripped)
    in more than 30 % of the total pages and has ≤ 8 words.

    Parameters
    ----------
    pages_text:
        Raw text of every page as a list.

    Returns
    -------
    list[str]
        The same list with suspected headers/footers removed.
    """
    if not pages_text:
        return pages_text

    total_pages = len(pages_text)
    threshold = max(2, int(total_pages * 0.30))

    # Count per-line occurrences across all pages (first + last 3 lines only)
    line_counts: dict[str, int] = {}
    for page in pages_text:
        lines = page.splitlines()
        boundary_lines = lines[:3] + lines[-3:]
        for ln in boundary_lines:
            stripped = ln.strip()
            if stripped and len(stripped.split()) <= 8:
                line_counts[stripped] = line_counts.get(stripped, 0) + 1

    # Identify recurring headers/footers
    recurring = {ln for ln, cnt in line_counts.items() if cnt >= threshold}
    if recurring:
        logger.debug("Detected %d recurring header/footer patterns.", len(recurring))

    cleaned: list[str] = []
    for page in pages_text:
        filtered_lines = [
            ln for ln in page.splitlines() if ln.strip() not in recurring
        ]
        cleaned.append("\n".join(filtered_lines))
    return cleaned


class PDFExtractor:
    """
    Extracts and cleans text from a novel PDF using *pdfplumber*.

    Parameters
    ----------
    pdf_path:
        Absolute path to the input PDF file.
    """

    def __init__(self, pdf_path: Path) -> None:
        self.pdf_path = pdf_path
        self._raw_pages: list[str] = []
        self._clean_text: str = ""

    # ─── Public API ───────────────────────────────────────────────────────

    def extract(self) -> str:
        """
        Run the full extraction pipeline.

        Returns
        -------
        str
            The cleaned, continuous plain-text body of the novel.
        """
        logger.info("Extracting PDF: %s", self.pdf_path)
        self._read_pages()
        self._clean_pages()
        self._assemble()
        logger.info(
            "Extraction complete. Total characters: %d", len(self._clean_text)
        )
        return self._clean_text

    # ─── Private helpers ──────────────────────────────────────────────────

    def _read_pages(self) -> None:
        """Open the PDF with pdfplumber and collect per-page raw text."""
        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                total = len(pdf.pages)
                logger.info("PDF has %d pages.", total)
                for i, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text(x_tolerance=2, y_tolerance=2) or ""
                    self._raw_pages.append(text)
                    if i % 50 == 0:
                        logger.debug("  … read %d / %d pages", i, total)
        except Exception as exc:
            raise RuntimeError(f"Failed to read PDF '{self.pdf_path}': {exc}") from exc

    def _clean_pages(self) -> None:
        """Apply all cleaning steps to the raw page list."""
        # Step 1 – Remove recurring headers/footers
        pages = _remove_headers_footers(self._raw_pages)

        # Step 2 – Per-page noise removal
        cleaned: list[str] = []
        for page_text in pages:
            lines = page_text.splitlines()
            good_lines: list[str] = []
            for line in lines:
                if _is_noise_line(line):
                    continue
                # Fix broken OCR spacing: remove mid-word hyphen breaks
                line = re.sub(r"(\w)-\n(\w)", r"\1\2", line)
                good_lines.append(line)
            cleaned.append("\n".join(good_lines))

        self._raw_pages = cleaned

    def _assemble(self) -> None:
        """Concatenate pages, normalise Unicode and whitespace."""
        full_text = "\n\n".join(self._raw_pages)
        full_text = normalise_unicode(full_text)
        full_text = clean_whitespace(full_text)

        # Collapse excessive blank lines (3+ → 2) — clean_whitespace handles this
        self._clean_text = full_text


def extract_pdf(pdf_path: Path) -> str:
    """
    Convenience function: extract and clean text from *pdf_path*.

    Parameters
    ----------
    pdf_path:
        Path to the input PDF.

    Returns
    -------
    str
        Clean novel text.
    """
    extractor = PDFExtractor(pdf_path)
    return extractor.extract()
