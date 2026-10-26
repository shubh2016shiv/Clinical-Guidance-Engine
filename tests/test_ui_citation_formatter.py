"""Citations belong under the answer, listing each source once."""

import unittest

from healthcare_assistant_ui.citation_formatter import ChainlitCitationFormatter


def citation(filename="diabetes_guidelines.md", **fields):
    return {"filename": filename, "file_id": "file_1", "index": 3, **fields}


class CitationsSectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.formatter = ChainlitCitationFormatter()

    def test_sources_are_listed_under_a_references_heading(self) -> None:
        section = self.formatter.format_citations_section([citation()])

        self.assertIn("### References", section)
        self.assertIn("[1] diabetes_guidelines.md", section)

    def test_sources_are_numbered_in_the_order_given_to_match_the_inline_markers(self) -> None:
        section = self.formatter.format_citations_section([citation("b.md"), citation("a.md")])

        self.assertIn("[1] b.md", section)
        self.assertIn("[2] a.md", section)
        self.assertNotIn("[3]", section)

    def test_a_long_quote_is_truncated(self) -> None:
        section = self.formatter.format_citations_section([citation(quote="x" * 200)])

        self.assertIn("...", section)
        self.assertNotIn("x" * 160, section)

    def test_no_citations_adds_nothing_to_the_answer(self) -> None:
        self.assertEqual(self.formatter.format_citations_section([]), "")

    def test_the_formatter_no_longer_builds_side_panel_elements(self) -> None:
        """The references render in the message; a side panel would duplicate them."""
        self.assertFalse(hasattr(self.formatter, "create_citation_elements"))


if __name__ == "__main__":
    unittest.main()
