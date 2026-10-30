"""Place numbered file citation markers at their offsets in a guideline answer.

The file search tells us WHICH guideline file backed a claim and WHERE in
the answer text that claim sits (a character offset). This file turns that
raw information into something a reader can follow: a small raised marker
in the text, plus a numbered source list to match.

Big-picture flow:
1. Split the citations into those that know their offset and those that do not.
2. Walk the ones with offsets in text order, giving each distinct file its
   own source number (a file cited five times keeps one number).
3. Give the offset-less citations numbers too, so their sources are still
   listed, just without a marker in the text.
4. Insert the markers into the text working BACKWARDS from the end, so
   earlier offsets are not shifted by text inserted after them.
5. Return the marked-up text together with the numbered source list.
"""

# re is Python's regular-expression library: it finds and replaces text
# matching a pattern, used here to strip markers that point at a source
# the answer does not actually have.
# @dataclass writes the boring parts of a class (like __init__) from the
# fields declared below it.
# The ".." means "go up one folder" - FileCitation records one citation:
# which file, its name, and the character offset it belongs at.
import re
from dataclasses import dataclass

from ..schema import FileCitation

_INLINE_CITATION_MARKER_PATTERN = re.compile(r"\$\{\}\^\{\[(\d+)\]\}\$")
# This pattern finds an already-inserted marker such as ${}^{[1]}$. Every
# "$", "{", "}" and "^" is written as "\$", "\{" ... because those
# characters mean something special to the regex engine, and the backslash
# says "match the literal character instead". "(\d+)" captures the digits
# so the code can read back WHICH source number a marker refers to.
# A leading underscore marks these as private to this file.
# A marker belongs after the punctuation that ends the cited sentence, not before it.
_PUNCTUATION_BEFORE_CITATION_MARKER = ".,;:!?"


# CitationMarkedText: the pair of results this file produces - the answer
# text with markers inserted, and the sources those marker numbers refer
# to, in numbered order. Needed because the two only make sense together:
# a marker reading "1" is meaningless without the list saying what 1 is.
@dataclass(frozen=True)
class CitationMarkedText:
    """Answer text with inline markers, and the sources those markers number."""

    # "frozen=True" above makes instances read-only once created, so a
    # caller cannot accidentally change the text and leave the numbers
    # pointing at the wrong sources.
    text: str
    # text: the answer with its citation markers inserted.
    sources: list[FileCitation]
    # sources: the cited files in numbered order, so the file at position 0
    # is source number 1, position 1 is source number 2, and so on.


# format_inline_citation_marker: builds the visible marker for one source
# number. Kept in its own function so the marker style is defined in a
# single place, and so the pattern above and the output here cannot drift
# apart unnoticed.
def format_inline_citation_marker(source_number: int) -> str:
    """Return a math marker that raises both square brackets and the source number."""
    # The "$...$" wrapper asks the frontend to render this as inline maths.
    # "{}^{...}" is a superscript attached to an empty base, which is the
    # trick that lifts the whole "[1]" - brackets included - rather than
    # just the digit.
    # NOTE: this renders as a raised "[1]" only where maths is supported
    # (the web UI). Somewhere plain, such as terminal output, the
    # reader sees the literal text ${}^{[1]}$.
    return "${}^{[" + str(source_number) + "]}$"


# insert_file_citation_markers: the main job of this file. Needed because
# the file search hands back citations in its own order, possibly several
# for the same file and several at the same spot, and none of that is
# readable until it has been numbered, de-duplicated and placed.
def insert_file_citation_markers(
    answer_text: str, file_citations: list[FileCitation]
) -> CitationMarkedText:
    """Insert a marker at each citation offset and number the sources by first appearance.

    ``FileCitation.index`` is the character offset in ``answer_text`` where the file search
    attached the citation. A source cited several times keeps one number. A citation with
    no offset cannot be placed, so its source is listed after the placed ones, unmarked.
    """
    # Step 1: separate the citations that know where they belong from those
    # that do not, and put the located ones in text order. Sorting by
    # offset is what makes the numbering follow the order a READER meets
    # the citations, rather than the order the search happened to return.
    # "(citation for citation in ...)" is a generator expression: it feeds
    # items to sorted() one at a time instead of building a throwaway list.
    # "key=lambda citation: citation.index" is a tiny unnamed function
    # telling sorted() to compare citations by their offset.
    citations_with_offsets = sorted(
        (citation for citation in file_citations if citation.index is not None),
        key=lambda citation: citation.index,
    )
    citations_without_offsets = [citation for citation in file_citations if citation.index is None]
    # citations_without_offsets: cited files with no position, collected
    # with a list comprehension because the whole list is wanted at once.

    numbered_file_sources: list[FileCitation] = []
    # numbered_file_sources: one entry per distinct cited file, in the
    # order first met. Its length doubles as the next number to hand out.
    source_numbers_by_file_id: dict[str, int] = {}
    # source_numbers_by_file_id: remembers the number already given to a
    # file, so the same file cited again reuses its number instead of
    # taking a new one.
    source_numbers_by_text_offset: dict[int, list[int]] = {}
    # source_numbers_by_text_offset: which source numbers to print at which
    # offset. The value is a LIST because one spot in the text can cite
    # several different sources at once.
    # Step 2: number the located citations, and note where each marker goes.
    for file_citation in citations_with_offsets:
        if not 0 <= file_citation.index <= len(answer_text):
            # An offset outside the text would silently put the marker in
            # the wrong place (or crash the slicing later), so refuse it
            # now with a message naming both numbers. "0 <= x <= len" is a
            # chained comparison, Python's way of writing two checks at once.
            # The end of the text counts as valid, which is why it is "<=".
            raise ValueError(
                f"Citation offset {file_citation.index} is outside the {len(answer_text)}-character answer"
            )
        if file_citation.file_id not in source_numbers_by_file_id:
            # First time this file is cited: add it to the source list and
            # give it the next number. Numbers are 1-based for readers, so
            # the number is the list length AFTER appending.
            numbered_file_sources.append(file_citation)
            source_numbers_by_file_id[file_citation.file_id] = len(numbered_file_sources)
        marker_offset = _citation_offset_after_punctuation(answer_text, file_citation.index)
        # marker_offset: where the marker will actually be written, nudged
        # past a full stop or comma if the citation landed right on one.
        source_numbers_at_offset = source_numbers_by_text_offset.setdefault(marker_offset, [])
        # ".setdefault(key, [])" returns the list already stored for this
        # offset, or puts an empty list there and returns that. It saves
        # writing "if the key is missing, create it first" by hand - and
        # because it returns the SAME list that is in the dictionary,
        # appending below updates the dictionary too.
        if source_numbers_by_file_id[file_citation.file_id] not in source_numbers_at_offset:
            # Guard against printing the same number twice in one spot,
            # which would read as "[1][1]".
            source_numbers_at_offset.append(source_numbers_by_file_id[file_citation.file_id])

    # Step 3: give the offset-less citations numbers as well. They get no
    # marker in the text because there is nowhere to put one, but their
    # files still belong in the source list so the reader can see them.
    for file_citation in citations_without_offsets:
        if file_citation.file_id not in source_numbers_by_file_id:
            numbered_file_sources.append(file_citation)
            source_numbers_by_file_id[file_citation.file_id] = len(numbered_file_sources)

    text_with_citation_markers = answer_text
    # text_with_citation_markers: the answer being built up, starting as
    # the original text and gaining one group of markers per offset below.
    # Step 4: insert the markers from the LAST offset to the first.
    # "sorted(a_dictionary)" walks its keys, and "reverse=True" walks them
    # largest first. Going backwards is essential: inserting characters
    # pushes everything after them further along, so working forwards would
    # leave every later offset pointing at the wrong character. Offsets
    # before the insertion point are unaffected, so backwards is safe.
    for marker_offset in sorted(source_numbers_by_text_offset, reverse=True):
        citation_markers = "".join(
            format_inline_citation_marker(source_number)
            for source_number in source_numbers_by_text_offset[marker_offset]
        )
        # citation_markers: every marker for this one spot, run together as
        # a single string, e.g. two sources cited together.
        text_with_citation_markers = (
            text_with_citation_markers[:marker_offset]
            + citation_markers
            + text_with_citation_markers[marker_offset:]
        )
        # Slicing splits the text at the offset and rebuilds it with the
        # markers wedged in between: everything before, the markers, then
        # everything from the offset onward.
    # Step 5: hand back the marked text and the numbered sources together.
    return CitationMarkedText(text=text_with_citation_markers, sources=numbered_file_sources)


# remove_unknown_file_citation_markers: deletes markers that point at a
# source number the answer does not have. Needed because the model can
# write a marker of its own accord, and a "[4]" with only three sources
# listed would send the reader looking for something that is not there.
def remove_unknown_file_citation_markers(answer_text: str, available_source_count: int) -> str:
    """Remove markers that number a source the answer does not have."""

    def keep_marker_for_known_source(marker_match: re.Match[str]) -> str:
        # This small function is handed to .sub() below, which calls it once
        # per marker it finds and puts the returned text in that marker's
        # place. It can read "available_source_count" from the enclosing
        # function without it being passed in.
        source_number = int(marker_match.group(1))
        # ".group(1)" is the digits captured by the "(\d+)" part of the
        # pattern - the source number as text, converted to a whole number.
        return marker_match.group(0) if 1 <= source_number <= available_source_count else ""
        # ".group(0)" is the whole marker exactly as found, so a valid
        # marker is put back unchanged. Returning "" instead deletes it.

    return _INLINE_CITATION_MARKER_PATTERN.sub(keep_marker_for_known_source, answer_text)


# _citation_offset_after_punctuation: moves a marker one character to the
# right when the citation landed on sentence-ending punctuation, so the
# result reads "...guideline.[1]" rather than "...guideline[1]."
def _citation_offset_after_punctuation(answer_text: str, citation_offset: int) -> int:
    if (
        citation_offset < len(answer_text)
        and answer_text[citation_offset] in _PUNCTUATION_BEFORE_CITATION_MARKER
    ):
        # The length check has to come first: an offset can legitimately sit
        # at the very end of the text, where there is no character to look
        # at, and reading it would raise an error. Python stops evaluating
        # an "and" as soon as the left side is false, which is what makes
        # ordering the checks this way safe.
        return citation_offset + 1
    return citation_offset
