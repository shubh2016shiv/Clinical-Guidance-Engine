"""Helper for asking the chat model a question and getting back a typed Python object.

Two of the graph steps (deciding the route and extracting drug classes) need the
model to answer in a fixed JSON shape. This file holds that shared logic.
"""

from typing import Any

# Very low temperature = the model gives almost the same answer every time.
TEMPERATURE = 0.01


async def ask_model_for_structured_json(
    chat_completions_provider: Any,
    model_name: str,
    system_prompt: str,
    user_content: str,
    output_model: type[Any],
    previous_conversation_messages: list[dict[str, str]] | None = None,
) -> Any:
    """Ask the model a question and force its reply into a fixed JSON shape.

    `output_model` describes the shape we want. We send that shape to the model,
    then parse the reply into that same class.

    `previous_conversation_messages` are earlier user/assistant messages. They go
    between the system prompt and the current question, in the order they happened,
    so the model reads the conversation like a transcript.
    """
    result = await chat_completions_provider.execute_chat_completion(
        [
            {"role": "system", "content": system_prompt},
            *(previous_conversation_messages or []),
            {"role": "user", "content": user_content},
        ],
        model=model_name,
        temperature=TEMPERATURE,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": output_model.__name__,
                "schema": output_model.model_json_schema(),
                "strict": True,
            },
        },
    )

    # Safety checks: the model must have finished normally and actually replied.
    if result.finish_reason != "stop":
        raise RuntimeError(f"{output_model.__name__} ended early: {result.finish_reason}")
    if not result.content:
        raise RuntimeError(f"{output_model.__name__} had no text; the model may have refused")

    return output_model.model_validate_json(result.content)
