"""
Citation Formatter for Chainlit UI.

Handles formatting and display of citations from clinical guidelines and drug databases,
supporting both inline markdown citations and expandable side elements.
"""

import re
from typing import List, Dict, Any, Optional
import chainlit as cl


class ChainlitCitationFormatter:
    """
    Formats citations for display in the Chainlit UI.

    Supports multiple display modes:
    1. Inline citations with markdown footnotes
    2. Expandable side elements showing citation details
    3. Formatted citation section appended to response
    """

    def __init__(self):
        """Initialize the citation formatter."""
        self.citation_pattern = re.compile(r"\[(\d+)\]")

    def format_inline_citations(
        self, text: str, citations: List[Dict[str, Any]]
    ) -> str:
        """
        Format inline citations in the response text with markdown links.

        Replaces citation markers like [1], [2] with markdown footnote references.

        Args:
            text: The response text containing citation markers.
            citations: List of citation dictionaries with metadata.

        Returns:
            Text with formatted inline citations.
        """
        if not citations:
            return text

        # Create a mapping of citation indices to citation data
        citation_map = {i + 1: citation for i, citation in enumerate(citations)}

        def replace_citation(match):
            citation_num = int(match.group(1))
            if citation_num in citation_map:
                # Format as markdown footnote
                return f"[^{citation_num}]"
            return match.group(0)

        # Replace citation markers with markdown footnotes
        formatted_text = self.citation_pattern.sub(replace_citation, text)

        return formatted_text

    def create_citation_elements(
        self, citations: List[Dict[str, Any]]
    ) -> List[cl.Text]:
        """
        Create Chainlit Text elements for citations to display as side elements.

        Each citation becomes an expandable element showing the source filename
        and relevant quote from the document.

        Args:
            citations: List of citation dictionaries with metadata.

        Returns:
            List of Chainlit Text elements for display.
        """
        if not citations:
            return []

        citation_elements = []

        for idx, citation in enumerate(citations, start=1):
            filename = citation.get("filename", "Unknown Source")
            quote = citation.get("quote", "No quote available")
            file_id = citation.get("file_id", "")

            # Format citation content
            content = f"**Source:** {filename}\n\n"

            if quote:
                content += f"**Excerpt:**\n> {quote}\n\n"

            if file_id:
                content += f"*File ID: {file_id}*"

            # Create Text element
            element = cl.Text(
                name=f"Citation {idx}",
                content=content,
                display="side",
            )

            citation_elements.append(element)

        return citation_elements

    def format_citation_text(self, citation: Dict[str, Any], index: int) -> str:
        """
        Format a single citation as a text string.

        Args:
            citation: Citation dictionary with metadata.
            index: Citation number (1-indexed).

        Returns:
            Formatted citation string.
        """
        filename = citation.get("filename", "Unknown Source")
        quote = citation.get("quote", "")

        citation_str = f"[{index}] {filename}"

        if quote:
            # Truncate long quotes
            max_quote_length = 150
            if len(quote) > max_quote_length:
                quote = quote[:max_quote_length] + "..."
            citation_str += f'\n    "{quote}"'

        return citation_str

    def format_citations_section(self, citations: List[Dict[str, Any]]) -> str:
        """
        Format all citations as a markdown section to append to the response.

        Creates a "References" section with numbered citations.

        Args:
            citations: List of citation dictionaries with metadata.

        Returns:
            Formatted citations section as markdown string.
        """
        if not citations:
            return ""

        section_lines = ["\n\n---\n\n### References\n"]

        for idx, citation in enumerate(citations, start=1):
            citation_text = self.format_citation_text(citation, idx)
            section_lines.append(f"\n{citation_text}\n")

        return "".join(section_lines)

    def extract_citations_from_metadata(
        self, metadata: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Extract citation information from response metadata.

        Args:
            metadata: Response metadata that may contain citations.

        Returns:
            List of extracted citation dictionaries.
        """
        if not metadata:
            return []

        # Check for citations in common metadata locations
        citations = metadata.get("citations", [])

        if not citations:
            # Check alternative locations
            citations = metadata.get("annotations", [])

        return citations if isinstance(citations, list) else []

    def format_combined_citations(
        self, text: str, citations: List[Dict[str, Any]]
    ) -> tuple[str, List[cl.Text]]:
        """
        Format citations for both inline display and side elements.

        Combines inline formatting and creates side elements for comprehensive
        citation display.

        Args:
            text: Response text containing citation markers.
            citations: List of citation dictionaries with metadata.

        Returns:
            Tuple of (formatted_text_with_inline_citations, citation_elements).
        """
        # Format inline citations
        formatted_text = self.format_inline_citations(text, citations)

        # Add citations section at the end
        formatted_text += self.format_citations_section(citations)

        # Create side elements
        citation_elements = self.create_citation_elements(citations)

        return formatted_text, citation_elements
