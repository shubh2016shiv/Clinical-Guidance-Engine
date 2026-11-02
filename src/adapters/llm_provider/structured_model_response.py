"""Helper for asking the chat model a question and getting back a typed Python object.

Routing and guideline interpretation both request typed JSON replies. This helper
owns the shared provider request and validation, without importing either caller.
"""

from typing import Any, TypeVar

from pydantic import BaseModel

# ---------------------------------------------------------------------------
# MODEL SUPPORT — Keep the requested output type visible to callers
# ---------------------------------------------------------------------------
# Python/Pydantic output types describe this model reply, not the graph state schema.
OutputModel = TypeVar("OutputModel", bound=BaseModel)

# Preserve the existing sampling setting for these structured-response calls.
TEMPERATURE = 0.01


async def ask_model_for_structured_json(
    chat_completions_provider: Any,
    model_name: str,
    system_prompt: str,
    user_content: str,
    output_model: type[OutputModel],
    previous_conversation_messages: list[dict[str, str]] | None = None,
) -> OutputModel:
    """Request a strict JSON schema and return a validated instance of output_model.

    `output_model` describes the shape we want. We send that shape to the model,
    then parse the reply into that same class.

    `previous_conversation_messages` are earlier user/assistant messages. They go
    between the system prompt and the current question, in the order they happened,
    so the model reads the conversation like a transcript.

    Raise RuntimeError for an unfinished or empty reply. Invalid JSON and schema
    mismatches propagate as Pydantic validation errors; no retries or fallback values.
    """
    # ---------------------------------------------------------------------------
    # MODEL 1 — Instructions first, earlier messages next, current input last
    # ---------------------------------------------------------------------------
    messages = [
        {"role": "system", "content": system_prompt},
        *(previous_conversation_messages or []),
        {"role": "user", "content": user_content},
    ]
    # ---------------------------------------------------------------------------
    # MODEL 2 — Describe the exact JSON object expected from the provider
    # ---------------------------------------------------------------------------
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": output_model.__name__,
            "schema": output_model.model_json_schema(),
            "strict": True,
        },
    }

    # ---------------------------------------------------------------------------
    # MODEL 3 — Make one model call without retries or fallback values
    # ---------------------------------------------------------------------------
    model_reply = await chat_completions_provider.execute_chat_completion(
        messages,
        model=model_name,
        temperature=TEMPERATURE,
        response_format=response_format,
    )

    # ---------------------------------------------------------------------------
    # MODEL 4 — Reject unfinished/empty replies, then validate JSON and fields
    # ---------------------------------------------------------------------------
    # RuntimeError reports completion/content problems. Pydantic errors describe malformed
    # JSON or mismatched fields. Both propagate to the caller.
    if model_reply.finish_reason != "stop":
        raise RuntimeError(f"{output_model.__name__} ended early: {model_reply.finish_reason}")
    if not model_reply.content:
        raise RuntimeError(f"{output_model.__name__} had no text; the model may have refused")

    return output_model.model_validate_json(model_reply.content)
