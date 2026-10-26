"""Describe the shapes of requests sent to OpenAI.

These types help readers and editors see which fields each API accepts.
1. Describe the required fields for Chat Completions requests.
2. Describe the required fields for Responses API requests.
3. Mark optional fields so callers know what they may leave out.

Beginner note: nothing in this file actually runs when a request is sent.
It only exists so your editor and type checker can warn you if you forget a
required field or misspell one, like a blueprint checked against the final
building.
"""

# TypedDict describes dictionary fields for type checking; it does not change them at runtime.
# A TypedDict looks like a class, but when you actually use it, it behaves
# like a normal Python dictionary (curly braces {}). It exists purely so
# tools can check "does this dict have the right keys and value types?"
from typing import Any, NotRequired, TypedDict


# This request shape documents the messages and options sent to Chat Completions.
class ChatCompletionRequest(TypedDict):
    """Chat Completions SDK request parameters."""

    model: str
    # model: which OpenAI model name to call.
    messages: list[dict[str, str]]
    # messages: the conversation so far, as a list of {"role": ..., "content": ...}.
    stream: bool
    # stream: whether to receive the answer in pieces instead of all at once.
    # NotRequired means the caller may omit these settings and use API defaults.
    temperature: NotRequired[float]
    top_p: NotRequired[float]
    # max_completion_tokens replaces the deprecated max_tokens; newer models reject the old name.
    max_completion_tokens: NotRequired[int]
    frequency_penalty: NotRequired[float]
    presence_penalty: NotRequired[float]
    stop: NotRequired[str | list[str]]


# The Responses API accepts input and instructions in a different shape.
class ResponsesApiRequest(TypedDict):
    """Responses API SDK request parameters."""

    model: str
    # model: which OpenAI model name to call.
    input: str | list[dict[str, Any]]
    # input: either plain text, or a structured list of input items.
    stream: bool
    # stream: whether to receive the answer in pieces instead of all at once.
    # The remaining fields are optional parts of a Responses API request.
    instructions: NotRequired[str]
    previous_response_id: NotRequired[str]
    tools: NotRequired[list[dict[str, Any]]]
    max_output_tokens: NotRequired[int]
