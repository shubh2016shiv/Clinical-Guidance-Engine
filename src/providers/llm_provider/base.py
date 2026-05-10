"""
Abstract LLM Provider Interface.

Defines the contract that every LLM provider implementation must fulfil.
The interface supports multiple API paradigms (chat-completion, responses,
etc.), streaming and non-streaming modes, and token-budget validation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from src.logs import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class APIType(Enum):
    """Supported LLM API paradigms."""

    CHAT_COMPLETION = "chat_completion"
    RESPONSES = "responses"
    COMPLETION = "completion"  # legacy text-completion


class StreamMode(Enum):
    """Streaming behaviour for LLM responses."""

    DISABLED = "disabled"
    ENABLED = "enabled"
    AUTO = "auto"  # provider decides per-request


# ---------------------------------------------------------------------------
# Configuration & Response dataclasses
# ---------------------------------------------------------------------------


@dataclass
class LLMConfig:
    """Provider-agnostic configuration for an LLM provider."""

    # ---- Model selection ---------------------------------------------------
    model_name: str
    api_type: APIType = APIType.CHAT_COMPLETION
    token_limit: int = 8192

    # ---- Generation parameters ---------------------------------------------
    temperature: float = 0.7
    max_tokens: int | None = None
    top_p: float | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    stop: str | list[str] | None = None

    # ---- Streaming ---------------------------------------------------------
    stream_mode: StreamMode = StreamMode.DISABLED

    # ---- Resilience --------------------------------------------------------
    timeout: int = 60  # seconds
    max_retries: int = 3
    retry_delay: float = 1.0  # initial back-off in seconds

    # ---- Extension point ---------------------------------------------------
    extra_params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise config (excluding secrets) for logging."""
        return {
            "model_name": self.model_name,
            "api_type": self.api_type.value,
            "token_limit": self.token_limit,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream_mode": self.stream_mode.value,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
        }


@dataclass
class LLMResponse:
    """Normalised response returned by any LLM provider."""

    content: str
    model: str
    provider_type: str
    api_type: APIType

    # ---- Usage (tokens) ----------------------------------------------------
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None

    # ---- Metadata ----------------------------------------------------------
    finish_reason: str | None = None
    response_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise response for downstream consumers."""
        return {
            "content": self.content,
            "model": self.model,
            "provider_type": self.provider_type,
            "api_type": self.api_type.value,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "finish_reason": self.finish_reason,
            "response_id": self.response_id,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Abstract Provider
# ---------------------------------------------------------------------------


class LLMProvider(ABC):
    """Abstract base class for every LLM provider implementation.

    Subclasses must implement at least ``execute_chat_completion``.
    The ``execute_responses_api`` method is *optional* — providers that
    do not support it should raise
    :exc:`~.exceptions.LLMProviderUnsupportedOperationError`.
    """

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def __init__(self, config: LLMConfig, **kwargs: Any) -> None:
        """Initialise the provider.

        Args:
            config: Fully populated :class:`LLMConfig`.
            **kwargs: Provider-specific keyword arguments (e.g. ``api_key``,
                ``model_name``).  Subclasses may consume these in their own
                ``__init__`` before calling ``super().__init__(config)``.
        """
        self.config = config
        self.model_name = config.model_name
        self.token_limit = config.token_limit

        # Derive a stable, human-readable provider-type slug.
        cls_name = self.__class__.__name__
        if cls_name.endswith("Provider"):
            cls_name = cls_name[: -len("Provider")]
        self.provider_type = cls_name.lower()

        logger.info(
            "Initialised %s provider – model=%s api=%s stream=%s",
            self.provider_type,
            self.model_name,
            config.api_type.value,
            config.stream_mode.value,
        )

    # ------------------------------------------------------------------
    # Core abstract methods
    # ------------------------------------------------------------------

    @abstractmethod
    async def execute(
        self,
        prompt: str,
        system_prompt: str | None = None,
        *,
        stream: bool = False,
        **kwargs: Any,
    ) -> LLMResponse | AsyncIterator[str]:
        """Execute an LLM call, routing to the appropriate API internally.

        Args:
            prompt: The user prompt.
            system_prompt: Optional system-level instructions.
            stream: When ``True``, return an ``AsyncIterator[str]`` of
                content chunks.  When ``False`` (default), return a
                complete :class:`LLMResponse`.
            **kwargs: Additional provider-specific parameters.

        Returns:
            ``LLMResponse`` for non-streaming, ``AsyncIterator[str]``
            for streaming.
        """
        ...

    @abstractmethod
    async def execute_chat_completion(
        self,
        messages: list[dict[str, str]],
        *,
        stream: bool = False,
        **kwargs: Any,
    ) -> LLMResponse | AsyncIterator[str]:
        """Execute a chat-completion API call.

        Args:
            messages: Ordered list of message dicts with ``role`` and
                ``content`` keys.
            stream: When ``True``, return an async iterator of content
                chunks instead of a complete ``LLMResponse``.
            **kwargs: Additional provider-specific parameters.

        Returns:
            ``LLMResponse`` for non-streaming, ``AsyncIterator[str]``
            for streaming.
        """
        ...

    # ------------------------------------------------------------------
    # Optional / specialised methods
    # ------------------------------------------------------------------

    async def execute_responses_api(
        self,
        input_message: str,
        instructions: str | None = None,
        previous_response_id: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        *,
        stream: bool = False,
        **kwargs: Any,
    ) -> LLMResponse | AsyncIterator[str]:
        """Execute an OpenAI Responses-API call.

        The default implementation raises
        :exc:`~.exceptions.LLMProviderUnsupportedOperationError`.
        Providers that support the Responses API **must** override this
        method.

        .. note::

            This method is deliberately **not** abstract so that
            non-OpenAI providers are not forced to implement it.
        """
        from .exceptions import LLMProviderUnsupportedOperationError

        raise LLMProviderUnsupportedOperationError(
            f"{self.provider_type} does not support the Responses API."
        )

    # ------------------------------------------------------------------
    # Token helpers
    # ------------------------------------------------------------------

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Return a conservative token-count estimate for *text*.

        Uses ``tiktoken`` when available; falls back to the ``chars /
        4`` heuristic otherwise.  The fallback is intentionally
        conservative (it over-estimates) to avoid accidental context-
        window violations.
        """
        try:
            import tiktoken

            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except Exception:
            # Fallback: ~4 characters per token for English text.
            return max(1, len(text) // 4)

    def validate_token_limit(
        self,
        prompt: str,
        max_tokens: int | None = None,
        system_prompt: str | None = None,
    ) -> bool:
        """Return ``True`` when *prompt* + expected response fits within
        the model's token budget."""
        prompt_tokens = self.estimate_tokens(prompt)
        if system_prompt:
            prompt_tokens += self.estimate_tokens(system_prompt)

        # Resolve response allocation: explicit > config > remaining budget.
        if max_tokens is not None:
            response_tokens = max_tokens
        elif self.config.max_tokens is not None:
            response_tokens = self.config.max_tokens
        else:
            response_tokens = max(0, self.token_limit - prompt_tokens)

        total = prompt_tokens + response_tokens

        if total > self.token_limit:
            logger.warning(
                "Token limit exceeded: %d > %d (input=%d, response=%d)",
                total,
                self.token_limit,
                prompt_tokens,
                response_tokens,
            )
            return False

        logger.debug(
            "Token budget ok: %d / %d (input=%d, response=%d)",
            total,
            self.token_limit,
            prompt_tokens,
            response_tokens,
        )
        return True

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def update_config(self, **kwargs: Any) -> None:
        """Update one or more config values at runtime."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.debug("config updated: %s = %s", key, value)
            else:
                logger.warning("Ignored unknown config key: %s", key)

    def get_model_info(self) -> dict[str, Any]:
        """Return a snapshot of the provider's identity & configuration."""
        return {
            "provider_type": self.provider_type,
            "model_name": self.model_name,
            "api_type": self.config.api_type.value,
            "token_limit": self.token_limit,
            "stream_mode": self.config.stream_mode.value,
            "config": self.config.to_dict(),
        }

    def should_stream(self) -> bool:
        """Return ``True`` when the provider is configured to stream."""
        if self.config.stream_mode == StreamMode.ENABLED:
            return True
        if self.config.stream_mode == StreamMode.DISABLED:
            return False
        # AUTO — default to non-streaming.
        return False
