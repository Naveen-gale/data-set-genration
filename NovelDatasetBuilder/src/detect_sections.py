"""
detect_sections.py
------------------
Automatically detects chapters, scenes, acts, prologues, epilogues and
dialogue sections inside the extracted novel text.

If no explicit structural markers are found the text is segmented by
semantic paragraph clusters so that narrative boundaries are still respected.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────

@dataclass
class Section:
    """Represents one logical section (chapter, scene, act, etc.) of the novel."""

    title: str
    body: str
    section_type: str  # e.g. "chapter", "scene", "act", "dialogue", "prose"
    index: int = 0
    word_count: int = field(init=False)

    def __post_init__(self) -> None:
        self.word_count = len(self.body.split())


# ─────────────────────────────────────────────
# Regex patterns for structural markers
# ─────────────────────────────────────────────

# Matches chapter/act/scene headings (robust multi-format)
_STRUCTURAL_PATTERN = re.compile(
    r"""
    ^\s*                                   # optional leading whitespace
    (?:
        # "Chapter 1", "Chapter One", "CHAPTER I"
        (?:chapter|chap\.?|ch\.?)\s+[\dIVXLCivxlc]+[.:\-]?\s*.*
        |
        # "Act I", "ACT 2"
        (?:act)\s+[\dIVXLCivxlc]+[.:\-]?\s*.*
        |
        # "Scene 1", "Scene III"
        (?:scene)\s+[\dIVXLCivxlc]+[.:\-]?\s*.*
        |
        # Standalone "PROLOGUE", "EPILOGUE", "INTRODUCTION", "INTERLUDE"
        (?:prologue|epilogue|introduction|interlude|preface|foreword|afterword|coda)
        |
        # All-caps title lines: "THE BEGINNING" (2–8 words, all uppercase letters)
        [A-Z][A-Z\s\-\']{4,60}
    )
    \s*$                                   # optional trailing whitespace
    """,
    re.VERBOSE | re.IGNORECASE | re.MULTILINE,
)

# Tighter pattern: must start a line and contain a known keyword
_HEADING_STRICT = re.compile(
    r"^(?:chapter|act|scene|prologue|epilogue|part|book|volume|section)\b.*$",
    re.IGNORECASE | re.MULTILINE,
)


def _find_structural_breaks(text: str) -> list[tuple[int, int, str]]:
    """
    Locate all heading positions in *text*.

    Returns
    -------
    list[tuple[int, int, str]]
        Each tuple is (start_char, end_char, heading_text).
    """
    breaks: list[tuple[int, int, str]] = []
    for m in _HEADING_STRICT.finditer(text):
        heading = m.group(0).strip()
        # Skip headings that are just a single word with ≥ 5 uppercase chars
        # and look like an acronym – likely a word in all-caps mid-sentence
        if len(heading.split()) == 1 and heading.isupper() and len(heading) <= 4:
            continue
        breaks.append((m.start(), m.end(), heading))
    return breaks


def _classify_heading(heading: str) -> str:
    """Return a section type string based on the heading text."""
    h = heading.lower()
    if re.match(r"chapter|chap", h):
        return "chapter"
    if re.match(r"act\b", h):
        return "act"
    if re.match(r"scene\b", h):
        return "scene"
    if re.match(r"prologue", h):
        return "prologue"
    if re.match(r"epilogue", h):
        return "epilogue"
    if re.match(r"part\b|book\b|volume\b", h):
        return "part"
    return "section"


def _split_by_headings(text: str) -> list[Section]:
    """
    Split *text* on detected structural headings.

    Each section spans from one heading to the start of the next.
    """
    breaks = _find_structural_breaks(text)
    if not breaks:
        return []

    sections: list[Section] = []
    for i, (start, end, heading) in enumerate(breaks):
        body_start = end
        body_end = breaks[i + 1][0] if i + 1 < len(breaks) else len(text)
        body = text[body_start:body_end].strip()
        if len(body.split()) < 50:
            # Too short – likely a sub-heading with almost no content; skip
            continue
        sec = Section(
            title=heading,
            body=body,
            section_type=_classify_heading(heading),
            index=len(sections),
        )
        sections.append(sec)
        logger.debug("Section [%s] '%s' — %d words", sec.section_type, sec.title, sec.word_count)

    return sections


def _split_by_paragraphs(text: str, target_words: int = 2000) -> list[Section]:
    """
    Fallback: split *text* into paragraph-cluster sections of approximately
    *target_words* words each. Never splits in the middle of a paragraph.

    Parameters
    ----------
    text:
        The full novel text.
    target_words:
        Approximate word count per section.
    """
    paragraphs = re.split(r"\n{2,}", text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    sections: list[Section] = []
    current_parts: list[str] = []
    current_words = 0

    for para in paragraphs:
        para_words = len(para.split())
        current_parts.append(para)
        current_words += para_words

        if current_words >= target_words:
            body = "\n\n".join(current_parts)
            idx = len(sections)
            sec = Section(
                title=f"Passage {idx + 1}",
                body=body,
                section_type="prose",
                index=idx,
            )
            sections.append(sec)
            current_parts = []
            current_words = 0

    # Flush any remaining paragraphs
    if current_parts:
        body = "\n\n".join(current_parts)
        idx = len(sections)
        sec = Section(
            title=f"Passage {idx + 1}",
            body=body,
            section_type="prose",
            index=idx,
        )
        sections.append(sec)

    return sections


class SectionDetector:
    """
    Detects and returns logical sections from the cleaned novel text.

    Parameters
    ----------
    text:
        The full cleaned novel text produced by the PDF extractor.
    """

    def __init__(self, text: str) -> None:
        self.text = text

    def detect(self) -> list[Section]:
        """
        Run section detection.

        Tries heading-based splitting first; falls back to paragraph clustering
        if fewer than 2 headings are found.

        Returns
        -------
        list[Section]
            Ordered list of detected sections.
        """
        sections = _split_by_headings(self.text)

        if len(sections) < 2:
            logger.info(
                "Fewer than 2 structural headings found. "
                "Falling back to paragraph-cluster splitting."
            )
            sections = _split_by_paragraphs(self.text)
        else:
            logger.info("Detected %d structural sections.", len(sections))

        return sections


def detect_sections(text: str) -> list[Section]:
    """
    Convenience wrapper around :class:`SectionDetector`.

    Parameters
    ----------
    text:
        Cleaned novel text.

    Returns
    -------
    list[Section]
        Ordered list of detected sections.
    """
    detector = SectionDetector(text)
    return detector.detect()
