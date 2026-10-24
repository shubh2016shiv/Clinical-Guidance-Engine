"""Place numbered citation markers in an answer where the file search cited a source."""

import re
from dataclasses import dataclass

from ..schema import FileCitation

_SUPERSCRIPT_DIGITS = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
_DIGITS_FROM_SUPERSCRIPT = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
_MARKER_PATTERN = re.compile(r"\[([⁰¹²³⁴⁵⁶⁷⁸⁹]+)\]")
# A marker belongs after the punctuation that ends the cited sentence, not before it.
_PUNCTUATION_BEFORE_MARKER = ".,;:!?"


@dataclass(frozen=True)
class CitedText:
    """Answer text with inline markers, and the sources those markers number."""

    text: str
    sources: list[FileCitation]


def citation_marker(source_number: int) -> str:
    """Return the marker for a 1-based source number, such as ``[¹]``."""
    return "[" + str(source_number).translate(_SUPERSCRIPT_DIGITS) + "]"


def mark_citations_in_text(text: str, citations: list[FileCitation]) -> CitedText:
    """Insert a marker at each citation offset and number the sources by first appearance.

    ``FileCitation.index`` is the character offset in ``text`` where the file search
    attached the citation. A source cited several times keeps one number. A citation with
    no offset cannot be placed, so its source is listed after the placed ones, unmarked.
    """
    placed = sorted((c for c in citations if c.index is not None), key=lambda c: c.index)
    unplaced = [c for c in citations if c.index is None]

    sources: list[FileCitation] = []
    source_numbers: dict[str, int] = {}
    markers_by_position: dict[int, list[int]] = {}
    for citation in placed:
        if not 0 <= citation.index <= len(text):
            raise ValueError(
                f"Citation offset {citation.index} is outside the {len(text)}-character answer"
            )
        if citation.file_id not in source_numbers:
            sources.append(citation)
            source_numbers[citation.file_id] = len(sources)
        position = _position_after_sentence_punctuation(text, citation.index)
        position_numbers = markers_by_position.setdefault(position, [])
        if source_numbers[citation.file_id] not in position_numbers:
            position_numbers.append(source_numbers[citation.file_id])

    for citation in unplaced:
        if citation.file_id not in source_numbers:
            sources.append(citation)
            source_numbers[citation.file_id] = len(sources)

    marked_text = text
    for position in sorted(markers_by_position, reverse=True):
        markers = "".join(citation_marker(number) for number in markers_by_position[position])
        marked_text = marked_text[:position] + markers + marked_text[position:]
    return CitedText(text=marked_text, sources=sources)


def drop_unknown_citation_markers(text: str, source_count: int) -> str:
    """Remove markers that number a source the answer does not have."""

    def keep_known_marker(marker_match: re.Match[str]) -> str:
        source_number = int(marker_match.group(1).translate(_DIGITS_FROM_SUPERSCRIPT))
        return marker_match.group(0) if 1 <= source_number <= source_count else ""

    return _MARKER_PATTERN.sub(keep_known_marker, text)


def _position_after_sentence_punctuation(text: str, index: int) -> int:
    if index < len(text) and text[index] in _PUNCTUATION_BEFORE_MARKER:
        return index + 1
    return index
