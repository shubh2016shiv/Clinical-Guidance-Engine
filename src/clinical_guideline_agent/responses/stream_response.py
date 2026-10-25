"""Interpret typed Responses API streaming events."""

from collections.abc import AsyncIterator
from typing import Any

from ..schema import GuidelineStreamUpdate
from .parse_response import parse_responses_api_result_into_guideline_answer


async def stream_guideline_updates_from_responses_api_events(
    responses_api_events: Any,
) -> AsyncIterator[GuidelineStreamUpdate]:
    """Convert Responses API events into guideline updates."""
    response_id: str | None = None
    async for responses_api_event in responses_api_events:
        responses_api_event_type = getattr(responses_api_event, "type", None)
        if responses_api_event_type == "response.created":
            response_id = getattr(getattr(responses_api_event, "response", None), "id", None)
        elif responses_api_event_type == "response.output_text.delta":
            yield GuidelineStreamUpdate(
                type="text",
                text=getattr(responses_api_event, "delta", ""),
                response_id=response_id,
            )
        elif responses_api_event_type == "response.completed":
            guideline_answer = parse_responses_api_result_into_guideline_answer(
                responses_api_event.response
            )
            yield GuidelineStreamUpdate(
                type="completed",
                response_id=guideline_answer.response_id,
                answer=guideline_answer,
            )
            return
        elif responses_api_event_type == "error":
            # The standalone "error" event is a transport-level stream failure;
            # the SDK puts its message directly on the event.
            raise RuntimeError(f"Guideline response stream error: {responses_api_event.message}")
        elif responses_api_event_type == "response.failed":
            # response.failed carries the full failed Response; the reason is
            # nested under response.error, not on the event itself.
            response_error = responses_api_event.response.error
            raise RuntimeError(
                f"Guideline response failed ({response_error.code}): {response_error.message}"
            )
        elif responses_api_event_type == "response.incomplete":
            # response.incomplete carries the full Response; the reason is
            # nested under response.incomplete_details.reason.
            incomplete_reason = responses_api_event.response.incomplete_details.reason
            raise RuntimeError(f"Guideline response is incomplete: {incomplete_reason}")
    raise RuntimeError("Responses API stream ended without response.completed")
