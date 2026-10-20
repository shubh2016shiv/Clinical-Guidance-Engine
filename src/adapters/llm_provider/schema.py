"""Data models for OpenAI LLM provider requests and responses."""

from typing import Any, NotRequired, TypedDict


class ChatCompletionRequest(TypedDict):
    """Chat Completions SDK request parameters."""

    model: str
    messages: list[dict[str, str]]
    stream: bool
    temperature: NotRequired[float]
    top_p: NotRequired[float]
    max_tokens: NotRequired[int]
    frequency_penalty: NotRequired[float]
    presence_penalty: NotRequired[float]
    stop: NotRequired[str | list[str]]


class ResponsesApiRequest(TypedDict):
    """Responses API SDK request parameters."""

    model: str
    input: str | list[dict[str, Any]]
    stream: bool
    instructions: NotRequired[str]
    previous_response_id: NotRequired[str]
    tools: NotRequired[list[dict[str, Any]]]
    max_tokens: NotRequired[int]
