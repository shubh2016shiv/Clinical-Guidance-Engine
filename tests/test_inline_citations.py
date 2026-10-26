"""Markers must land where the file search cited the source, and match the References list."""

import unittest

from src.clinical_guideline_agent.citations.inline_citations import (
    format_inline_citation_marker,
    insert_file_citation_markers,
    remove_unknown_file_citation_markers,
)
from src.clinical_guideline_agent.schema import FileCitation


def citation(index, file_id="file_a", filename="a.md"):
    return FileCitation(file_id=file_id, filename=filename, index=index)


class MarkCitationsTests(unittest.TestCase):
    def test_a_marker_follows_the_sentence_it_supports(self) -> None:
        text = "Start metformin. Add a second agent."

        cited = insert_file_citation_markers(text, [citation(15)])

        self.assertEqual(cited.text, "Start metformin.${}^{[1]}$ Add a second agent.")

    def test_the_marker_goes_after_the_full_stop_not_before_it(self) -> None:
        """The offset points at the full stop; the marker belongs after it."""
        text = "Start metformin."

        self.assertEqual(
            insert_file_citation_markers(text, [citation(15)]).text, "Start metformin.${}^{[1]}$"
        )

    def test_an_offset_at_the_very_end_of_the_text_is_marked(self) -> None:
        text = "Start metformin"

        self.assertEqual(
            insert_file_citation_markers(text, [citation(15)]).text, "Start metformin${}^{[1]}$"
        )

    def test_several_citations_keep_their_positions_despite_earlier_insertions(self) -> None:
        text = "One. Two. Three."

        cited = insert_file_citation_markers(
            text, [citation(3), citation(8, "file_b", "b.md"), citation(15)]
        )

        self.assertEqual(cited.text, "One.${}^{[1]}$ Two.${}^{[2]}$ Three.${}^{[1]}$")

    def test_sources_are_numbered_by_first_appearance_in_the_text(self) -> None:
        """The citations arrive in model order, not text order."""
        text = "One. Two."

        cited = insert_file_citation_markers(text, [citation(8, "file_b", "b.md"), citation(3)])

        self.assertEqual([source.filename for source in cited.sources], ["a.md", "b.md"])
        self.assertEqual(cited.text, "One.${}^{[1]}$ Two.${}^{[2]}$")

    def test_one_source_cited_repeatedly_has_one_number_and_one_list_entry(self) -> None:
        cited = insert_file_citation_markers("One. Two.", [citation(3), citation(8)])

        self.assertEqual(cited.text, "One.${}^{[1]}$ Two.${}^{[1]}$")
        self.assertEqual(len(cited.sources), 1)

    def test_two_sources_at_one_position_sit_side_by_side(self) -> None:
        cited = insert_file_citation_markers("One.", [citation(3), citation(3, "file_b", "b.md")])

        self.assertEqual(cited.text, "One.${}^{[1]}$${}^{[2]}$")

    def test_the_same_source_twice_at_one_position_is_marked_once(self) -> None:
        self.assertEqual(
            insert_file_citation_markers("One.", [citation(3), citation(3)]).text, "One.${}^{[1]}$"
        )

    def test_numbers_above_nine_use_every_digit(self) -> None:
        self.assertEqual(format_inline_citation_marker(12), "${}^{[12]}$")

    def test_a_citation_without_an_offset_is_listed_but_not_marked(self) -> None:
        cited = insert_file_citation_markers(
            "One.", [citation(3), citation(None, "file_b", "b.md")]
        )

        self.assertEqual(cited.text, "One.${}^{[1]}$")
        self.assertEqual([source.filename for source in cited.sources], ["a.md", "b.md"])

    def test_an_offset_beyond_the_text_is_an_error_not_a_misplaced_marker(self) -> None:
        with self.assertRaisesRegex(ValueError, "offset 99 is outside"):
            insert_file_citation_markers("One.", [citation(99)])

    def test_text_without_citations_is_returned_unchanged(self) -> None:
        cited = insert_file_citation_markers("One.", [])

        self.assertEqual(cited.text, "One.")
        self.assertEqual(cited.sources, [])


class DropUnknownMarkersTests(unittest.TestCase):
    def test_markers_for_existing_sources_are_kept(self) -> None:
        self.assertEqual(
            remove_unknown_file_citation_markers("A.${}^{[1]}$ B.${}^{[2]}$", 2),
            "A.${}^{[1]}$ B.${}^{[2]}$",
        )

    def test_a_marker_for_a_source_that_does_not_exist_is_removed(self) -> None:
        self.assertEqual(
            remove_unknown_file_citation_markers("A.${}^{[1]}$ B.${}^{[3]}$", 2), "A.${}^{[1]}$ B."
        )

    def test_text_with_no_sources_loses_every_marker(self) -> None:
        self.assertEqual(remove_unknown_file_citation_markers("A.${}^{[1]}$", 0), "A.")

    def test_ordinary_brackets_are_left_alone(self) -> None:
        text = "See [1] and [note] for details."

        self.assertEqual(remove_unknown_file_citation_markers(text, 0), text)


if __name__ == "__main__":
    unittest.main()
