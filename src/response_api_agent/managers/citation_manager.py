"""
Citation Manager for OpenAI Responses API

Handles extraction and formatting of citations from file_search in Responses API.
Provides reliable citation processing for both streaming and non-streaming responses.
"""

from typing import Dict, Any, List, Optional
from src.logs import get_component_logger


class CitationManager:
    """
    Manages citation extraction and formatting from OpenAI Responses API.

    Extracts file source information from annotations in the response output
    and formats them as numbered references with a comprehensive References section.
    """

    def __init__(self, client=None):
        """Initialize the Citation Manager."""
        self.logger = get_component_logger("Citation")
        self.client = client  # OpenAI client if needed for additional operations

    async def extract_citations_from_response(
        self, response: Any
    ) -> List[Dict[str, str]]:
        """
        Extract citation information from Responses API response object.

        In the Responses API, citations are found in annotations within the message content,
        NOT in tool_calls. The response structure is:
        - response.output[0] = file_search tool call
        - response.output[1] = assistant message with annotations

        Args:
            response: Response object from OpenAI Responses API.

        Returns:
            List of citation dictionaries with file information.
        """
        citations = []

        try:
            self.logger.info(
                "Extracting citations from Responses API response",
                component="Citation",
                subcomponent="ExtractCitationsFromResponse",
            )

            # Navigate to the annotations in the response
            annotations = self._extract_annotations_from_response(response)

            if not annotations:
                self.logger.warning(
                    "No annotations found in response",
                    component="Citation",
                    subcomponent="ExtractCitationsFromResponse",
                )
                return citations

            # Process each annotation
            for i, annotation in enumerate(annotations):
                citation = self._create_citation_from_annotation(annotation, i + 1)
                if citation:
                    citations.append(citation)

            self.logger.info(
                "Citations extracted successfully",
                component="Citation",
                subcomponent="ExtractCitationsFromResponse",
                citation_count=len(citations),
            )

            return citations

        except Exception as e:
            self.logger.error(
                "Error extracting citations from response",
                component="Citation",
                subcomponent="ExtractCitationsFromResponse",
                error=str(e),
            )
            return []

    def _extract_annotations_from_response(self, response: Any) -> List[Any]:
        """
        Extract annotations from the response output.

        The annotations are typically in:
        - response.output[1].content[0].annotations (most common)
        - or response.output[1].annotations (alternative structure)

        Args:
            response: Response object from OpenAI API.

        Returns:
            List of annotation objects.
        """
        try:
            # Check if response has output attribute
            if not hasattr(response, "output"):
                self.logger.warning(
                    "Response does not have 'output' attribute",
                    component="Citation",
                    subcomponent="ExtractAnnotations",
                )
                return []

            # Get the output list
            output = response.output
            if not output or len(output) < 2:
                self.logger.warning(
                    "Response output does not have enough elements",
                    component="Citation",
                    subcomponent="ExtractAnnotations",
                    output_length=len(output) if output else 0,
                )
                return []

            # The assistant message is typically at index 1
            # (index 0 is usually the file_search tool call)
            assistant_message = output[1]

            # Try to get annotations from content[0].annotations
            if hasattr(assistant_message, "content") and assistant_message.content:
                if len(assistant_message.content) > 0:
                    content_item = assistant_message.content[0]
                    if hasattr(content_item, "annotations"):
                        self.logger.debug(
                            "Found annotations in content[0].annotations",
                            component="Citation",
                            subcomponent="ExtractAnnotations",
                            annotation_count=len(content_item.annotations),
                        )
                        return content_item.annotations

            # Fallback: Try to get annotations directly from message
            if hasattr(assistant_message, "annotations"):
                self.logger.debug(
                    "Found annotations in message.annotations",
                    component="Citation",
                    subcomponent="ExtractAnnotations",
                    annotation_count=len(assistant_message.annotations),
                )
                return assistant_message.annotations

            self.logger.warning(
                "Could not find annotations in expected locations",
                component="Citation",
                subcomponent="ExtractAnnotations",
            )
            return []

        except Exception as e:
            self.logger.error(
                "Error extracting annotations",
                component="Citation",
                subcomponent="ExtractAnnotations",
                error=str(e),
            )
            return []

    def _create_citation_from_annotation(
        self, annotation: Any, citation_number: int
    ) -> Optional[Dict[str, str]]:
        """
        Create a citation dictionary from an annotation object.

        Annotations in Responses API have the structure:
        - annotation.filename: The filename of the source
        - annotation.file_id: The file ID (optional)
        - annotation.text: The citation marker text (optional)

        Args:
            annotation: Annotation object from response.
            citation_number: Number for this citation.

        Returns:
            Citation dictionary or None if extraction fails.
        """
        try:
            citation = {"number": str(citation_number), "filename": "", "file_id": ""}

            # Extract filename - this is the primary field we need
            if hasattr(annotation, "filename"):
                citation["filename"] = str(annotation.filename)
            elif isinstance(annotation, dict) and "filename" in annotation:
                citation["filename"] = str(annotation["filename"])

            # Extract file_id if available
            if hasattr(annotation, "file_id"):
                citation["file_id"] = str(annotation.file_id)
            elif isinstance(annotation, dict) and "file_id" in annotation:
                citation["file_id"] = str(annotation["file_id"])

            # If we don't have a filename, try to construct one from file_id
            if not citation["filename"] and citation["file_id"]:
                citation["filename"] = f"Document {citation['file_id'][:8]}"
                self.logger.debug(
                    "Using file_id to construct filename",
                    component="Citation",
                    subcomponent="CreateCitation",
                    filename=citation["filename"],
                )

            # Only return citation if we have at least a filename
            if citation["filename"]:
                self.logger.debug(
                    "Citation created successfully",
                    component="Citation",
                    subcomponent="CreateCitation",
                    citation_number=citation_number,
                    filename=citation["filename"],
                )
                return citation
            else:
                self.logger.warning(
                    "Could not extract filename from annotation",
                    component="Citation",
                    subcomponent="CreateCitation",
                    annotation_type=type(annotation).__name__,
                )
                return None

        except Exception as e:
            self.logger.error(
                "Error creating citation from annotation",
                component="Citation",
                subcomponent="CreateCitation",
                error=str(e),
            )
            return None

    def format_citations_section(self, citations: List[Dict[str, str]]) -> str:
        """
        Format citations as a simple References section with index and filename.

        Args:
            citations: List of citation dictionaries.

        Returns:
            Formatted References section string.
        """
        if not citations:
            return ""

        try:
            self.logger.info(
                "Formatting citations section",
                component="Citation",
                subcomponent="FormatCitationsSection",
                citation_count=len(citations),
            )

            # Create References header
            references_text = "## References\n\n"

            # Add each citation - only number and filename
            seen_filenames = set()
            for citation in citations:
                # Avoid duplicate filenames
                if citation["filename"] not in seen_filenames:
                    ref_text = f"[{citation['number']}] {citation['filename']}"
                    references_text += ref_text + "\n"
                    seen_filenames.add(citation["filename"])

            self.logger.info(
                "Citations section formatted successfully",
                component="Citation",
                subcomponent="FormatCitationsSection",
                unique_sources=len(seen_filenames),
            )

            return references_text

        except Exception as e:
            self.logger.error(
                "Error formatting citations section",
                component="Citation",
                subcomponent="FormatCitationsSection",
                error=str(e),
            )
            return "## References\n\n[Error formatting citations]"

    def append_citations_to_content(
        self, content: str, citations: List[Dict[str, str]]
    ) -> str:
        """
        Append formatted citations to content.

        Args:
            content: Original content string.
            citations: List of citation dictionaries.

        Returns:
            Content with citations appended.
        """
        if not citations:
            return content

        try:
            self.logger.info(
                "Appending citations to content",
                component="Citation",
                subcomponent="AppendCitationsToContent",
                content_length=len(content),
                citation_count=len(citations),
            )

            # Format citations section
            citations_section = self.format_citations_section(citations)

            # Append to content with proper spacing
            if citations_section:
                result = content + "\n\n" + citations_section
            else:
                result = content

            self.logger.info(
                "Citations appended successfully",
                component="Citation",
                subcomponent="AppendCitationsToContent",
                final_length=len(result),
            )

            return result

        except Exception as e:
            self.logger.error(
                "Error appending citations to content",
                component="Citation",
                subcomponent="AppendCitationsToContent",
                error=str(e),
            )
            return content

    # Legacy method for backward compatibility
    async def extract_citations_from_tool_calls(
        self, tool_calls: List[Any]
    ) -> List[Dict[str, str]]:
        """
        Legacy method - tool_calls don't contain citation info in Responses API.

        This method is kept for backward compatibility but will log a warning.
        Use extract_citations_from_response() instead.

        Args:
            tool_calls: List of tool call objects (not used in Responses API).

        Returns:
            Empty list with warning logged.
        """
        self.logger.warning(
            "extract_citations_from_tool_calls() called but tool_calls don't contain "
            "citation information in Responses API. Use extract_citations_from_response() instead.",
            component="Citation",
            subcomponent="ExtractCitationsFromToolCalls",
        )
        return []
