"""
chunk_builder.py
----------------
Takes the list of detected sections and produces intelligent text chunks
suitable for LLM instruction-tuning dataset generation.

Rules
-----
- Chunk size: 1 200 – 2 500 words.
- Never splits in the middle of a sentence, dialogue exchange, or paragraph.
- Adjacent chunks share a small word overlap to preserve narrative context.
- Each chunk carries metadata (source section, chunk index, word count).
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from src.config import CHUNK_MAX_WORDS, CHUNK_MIN_WORDS, CHUNK_OVERLAP_WORDS
from src.detect_sections import Section

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────

@dataclass
class Chunk:
    """A single text chunk ready for LLM processing."""

    text: str
    chunk_index: int
    source_section: str
    section_type: str
    word_count: int = field(init=False)

    def __post_init__(self) -> None:
        self.word_count = len(self.text.split())


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _split_into_paragraphs(text: str) -> list[str]:
    """Split text on one or more blank lines, preserving each paragraph."""
    paragraphs = re.split(r"\n{2,}", text)
    return [p.strip() for p in paragraphs if p.strip()]


def _split_into_sentences(paragraph: str) -> list[str]:
    """
    Naïve but robust sentence splitter that handles common abbreviations.

    Returns each sentence as a string.
    """
    # Insert markers at sentence boundaries
    marked = re.sub(
        r"(?<![A-Z][a-z]\.)(?<!\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|etc|vs|Fig)\.)([.!?])\s+(?=[A-Z\"])",
        r"\1\n",
        paragraph,
    )
    sentences = [s.strip() for s in marked.splitlines() if s.strip()]
    return sentences if sentences else [paragraph]


class ChunkBuilder:
    """
    Builds a flat list of text chunks from detected novel sections.

    Parameters
    ----------
    sections:
        Ordered list of :class:`Section` objects.
    min_words:
        Minimum words per chunk (default from config).
    max_words:
        Maximum words per chunk (default from config).
    overlap_words:
        Number of words to carry forward as overlap (default from config).
    """

    def __init__(
        self,
        sections: list[Section],
        min_words: int = CHUNK_MIN_WORDS,
        max_words: int = CHUNK_MAX_WORDS,
        overlap_words: int = CHUNK_OVERLAP_WORDS,
    ) -> None:
        self.sections = sections
        self.min_words = min_words
        self.max_words = max_words
        self.overlap_words = overlap_words
        self._chunks: list[Chunk] = []

    # ─── Public API ───────────────────────────────────────────────────────

    def build(self) -> list[Chunk]:
        """
        Process all sections and return the final chunk list.

        Returns
        -------
        list[Chunk]
            Ordered list of text chunks.
        """
        global_index = 0
        overlap_tail: str = ""  # words carried over from the previous chunk

        for section in self.sections:
            section_chunks = self._chunk_section(
                section=section,
                overlap_tail=overlap_tail,
                start_index=global_index,
            )
            if section_chunks:
                # Carry the tail of the last chunk as overlap into the next section
                last_text = section_chunks[-1].text
                last_words = last_text.split()
                overlap_tail = " ".join(last_words[-self.overlap_words:]) if len(last_words) > self.overlap_words else last_text
                self._chunks.extend(section_chunks)
                global_index += len(section_chunks)
            else:
                overlap_tail = ""

        logger.info("Built %d chunks total.", len(self._chunks))
        return self._chunks

    # ─── Private helpers ──────────────────────────────────────────────────

    def _chunk_section(
        self,
        section: Section,
        overlap_tail: str,
        start_index: int,
    ) -> list[Chunk]:
        """
        Chunk a single section, prepending any overlap tail from the previous
        section.

        Never splits mid-paragraph; if a single paragraph exceeds max_words it
        is split at sentence boundaries.
        """
        paragraphs = _split_into_paragraphs(section.body)
        chunks: list[Chunk] = []

        current_parts: list[str] = []
        if overlap_tail:
            current_parts.append(f"[…] {overlap_tail}")
        current_words = len(overlap_tail.split()) if overlap_tail else 0

        for para in paragraphs:
            para_words = len(para.split())

            # A single paragraph is already larger than max — split by sentence
            if para_words > self.max_words:
                sentences = _split_into_sentences(para)
                for sent in sentences:
                    sent_words = len(sent.split())
                    if current_words + sent_words > self.max_words and current_words >= self.min_words:
                        chunk = self._make_chunk(
                            current_parts, start_index + len(chunks), section
                        )
                        chunks.append(chunk)
                        current_parts = []
                        current_words = 0
                    current_parts.append(sent)
                    current_words += sent_words
                continue

            # Adding this paragraph would exceed max_words → flush first
            if current_words + para_words > self.max_words and current_words >= self.min_words:
                chunk = self._make_chunk(
                    current_parts, start_index + len(chunks), section
                )
                chunks.append(chunk)
                current_parts = []
                current_words = 0

            current_parts.append(para)
            current_words += para_words

        # Flush remaining content
        if current_parts and current_words >= 50:  # ignore tiny leftovers
            chunk = self._make_chunk(
                current_parts, start_index + len(chunks), section
            )
            chunks.append(chunk)

        return chunks

    def _make_chunk(
        self,
        parts: list[str],
        index: int,
        section: Section,
    ) -> Chunk:
        """Assemble parts into a :class:`Chunk` object."""
        text = "\n\n".join(parts).strip()
        return Chunk(
            text=text,
            chunk_index=index,
            source_section=section.title,
            section_type=section.section_type,
        )


def build_chunks(sections: list[Section]) -> list[Chunk]:
    """
    Convenience wrapper: build chunks from *sections* using config defaults.

    Parameters
    ----------
    sections:
        Ordered list of detected novel sections.

    Returns
    -------
    list[Chunk]
        Ordered list of text chunks.
    """
    builder = ChunkBuilder(sections)
    return builder.build()
