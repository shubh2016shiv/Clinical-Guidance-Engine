"""Parse a completed Responses API result without assuming output order."""

from typing import Any

from ..citations.file_citations import extract_responses_api_file_citations
from ..schema import GuidelineAnswer


def parse_responses_api_result_into_guideline_answer(
    responses_api_result: Any,
) -> GuidelineAnswer:
    """Read answer text and citations from a completed Responses API result."""
    response_id = getattr(responses_api_result, "id", None)
    if not response_id:
        raise ValueError("Response is missing its ID")
    responses_api_status = getattr(responses_api_result, "status", "completed")
    if responses_api_status == "failed":
        response_error = responses_api_result.error
        raise RuntimeError(
            f"Response {response_id} failed ({response_error.code}): {response_error.message}"
        )
    if responses_api_status == "incomplete":
        incomplete_reason = responses_api_result.incomplete_details.reason
        raise RuntimeError(f"Response {response_id} is incomplete: {incomplete_reason}")
    if responses_api_status != "completed":
        raise RuntimeError(f"Response {response_id} ended with status {responses_api_status}")
    answer_text = getattr(responses_api_result, "output_text", None)
    if answer_text is None:
        answer_text_parts = [
            getattr(message_content, "text", "")
            for output_item in getattr(responses_api_result, "output", []) or []
            if getattr(output_item, "type", None) == "message"
            for message_content in getattr(output_item, "content", []) or []
            if getattr(message_content, "type", None) == "output_text"
        ]
        answer_text = "".join(answer_text_parts)
    return GuidelineAnswer(
        response_id=response_id,
        text=answer_text,
        citations=extract_responses_api_file_citations(responses_api_result),
    )
