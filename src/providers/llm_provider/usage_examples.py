"""
LLM Provider Usage Examples.

Demonstrates the canonical usage patterns for the LLM provider system.
Copy snippets into your application code as needed.

.. note::

    Because :func:`create_llm_provider` returns the abstract
    :class:`~.base.LLMProvider` type (whose methods have union return
    types), the examples use :func:`typing.cast` to narrow the type at
    the call site.  This is the recommended pattern for production code
    when you know the streaming mode at compile-time.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import cast

from .base import APIType, LLMConfig, LLMResponse, StreamMode
from .exceptions import LLMProviderError
from .factory import create_llm_provider, get_default_provider

# ============================================================================
# EXAMPLE 1: Basic Chat Completion
# ============================================================================


async def example_basic_chat() -> None:
    provider = create_llm_provider(provider_type="openai", model_name="gpt-4")

    response = cast(
        LLMResponse,
        await provider.execute(
            prompt="What is the capital of France?",
            system_prompt="You are a helpful geography assistant.",
        ),
    )

    print(f"Response: {response.content}")
    print(f"Tokens used: {response.total_tokens}")
    print(f"Model: {response.model}")


# ============================================================================
# EXAMPLE 2: Streaming Chat Completion
# ============================================================================


async def example_streaming_chat() -> None:
    provider = create_llm_provider(
        provider_type="openai",
        model_name="gpt-4",
        stream_mode=StreamMode.ENABLED,
    )

    stream = cast(
        AsyncIterator[str],
        await provider.execute(
            prompt="Write a short poem about coding",
            system_prompt="You are a creative poet.",
            stream=True,
        ),
    )

    print("Streaming response:")
    async for chunk in stream:
        print(chunk, end="", flush=True)
    print()


# ============================================================================
# EXAMPLE 3: Responses API (Non-Streaming)
# ============================================================================


async def example_responses_api() -> None:
    provider = create_llm_provider(
        provider_type="openai",
        model_name="gpt-4",
        api_type=APIType.RESPONSES,
        stream_mode=StreamMode.DISABLED,
    )

    response = cast(
        LLMResponse,
        await provider.execute_responses_api(
            input_message="Hello! I'm working on a Python project.",
            instructions=(
                "You are a helpful Python programming assistant. Be concise and practical."
            ),
        ),
    )

    print(f"Assistant: {response.content}")

    response2 = cast(
        LLMResponse,
        await provider.execute_responses_api(
            input_message="Can you suggest some best practices?",
            instructions=(
                "You are a helpful Python programming assistant. Be concise and practical."
            ),
            previous_response_id=response.response_id,
        ),
    )

    print(f"Assistant: {response2.content}")


# ============================================================================
# EXAMPLE 4: Responses API (Streaming)
# ============================================================================


async def example_responses_api_streaming() -> None:
    provider = create_llm_provider(
        provider_type="openai",
        model_name="gpt-4",
        api_type=APIType.RESPONSES,
        stream_mode=StreamMode.ENABLED,
    )

    stream = cast(
        AsyncIterator[str],
        await provider.execute_responses_api(
            input_message="Explain async programming in Python",
            instructions="You are an expert Python developer. Explain concepts clearly.",
            stream=True,
        ),
    )

    print("Streaming response:")
    full_response: list[str] = []
    async for chunk in stream:
        print(chunk, end="", flush=True)
        full_response.append(chunk)

    print(f"\n\nTotal length: {len(''.join(full_response))} characters")


# ============================================================================
# EXAMPLE 5: Advanced Configuration with LLMConfig
# ============================================================================


async def example_advanced_config() -> None:
    config = LLMConfig(
        model_name="gpt-4-turbo",
        api_type=APIType.CHAT_COMPLETION,
        token_limit=128000,
        temperature=0.8,
        max_tokens=2000,
        top_p=0.9,
        frequency_penalty=0.5,
        presence_penalty=0.3,
        stop=["\n\n", "END"],
        stream_mode=StreamMode.DISABLED,
        timeout=120,
        max_retries=3,
        retry_delay=2.0,
    )

    provider = create_llm_provider(provider_type="openai", config=config)

    response = cast(LLMResponse, await provider.execute(prompt="Write a creative story about AI"))

    print(f"Response: {response.content}")
    print(f"Configuration used: {provider.config.to_dict()}")


# ============================================================================
# EXAMPLE 6: Dynamic Configuration Updates
# ============================================================================


async def example_dynamic_config() -> None:
    provider = create_llm_provider(provider_type="openai", model_name="gpt-4", temperature=0.7)

    response1 = cast(
        LLMResponse, await provider.execute(prompt="Generate a random number between 1 and 10")
    )
    print(f"Response 1: {response1.content}")

    provider.update_config(temperature=1.0, max_tokens=50)

    response2 = cast(
        LLMResponse, await provider.execute(prompt="Generate a random number between 1 and 10")
    )
    print(f"Response 2: {response2.content}")


# ============================================================================
# EXAMPLE 7: Error Handling
# ============================================================================


async def example_error_handling() -> None:
    try:
        provider = create_llm_provider(provider_type="openai", model_name="gpt-4")

        long_prompt = "A" * 200000  # exceeds token limit

        try:
            response = cast(LLMResponse, await provider.execute(prompt=long_prompt))
            print(f"Unexpected success: {response.content[:100]}...")
        except LLMProviderError:
            # Expected — token limit exceeded.
            pass

    except LLMProviderError as exc:
        print(f"Provider error: {exc}")
    except Exception as exc:
        print(f"Unexpected error: {exc}")


# ============================================================================
# EXAMPLE 8: Provider Information
# ============================================================================


async def example_provider_info() -> None:
    provider = create_llm_provider(
        provider_type="openai",
        model_name="gpt-4",
        api_type=APIType.RESPONSES,
        stream_mode=StreamMode.ENABLED,
    )

    info = provider.get_model_info()
    print("Provider Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")

    is_valid = provider.validate_token_limit("This is a test prompt", max_tokens=1000)
    print(f"\nToken validation passed: {is_valid}")


# ============================================================================
# EXAMPLE 9: Using the Default Provider
# ============================================================================


async def example_default_provider() -> None:
    provider = get_default_provider()
    response = cast(LLMResponse, await provider.execute(prompt="What is machine learning?"))
    print(f"Response: {response.content}")


# ============================================================================
# Main
# ============================================================================


async def main() -> None:
    print("=" * 80)
    print("EXAMPLE 1: Basic Chat Completion")
    print("=" * 80)
    await example_basic_chat()

    print("\n" + "=" * 80)
    print("EXAMPLE 2: Streaming Chat")
    print("=" * 80)
    await example_streaming_chat()

    print("\n" + "=" * 80)
    print("EXAMPLE 3: Responses API")
    print("=" * 80)
    await example_responses_api()

    print("\n" + "=" * 80)
    print("EXAMPLE 8: Provider Information")
    print("=" * 80)
    await example_provider_info()

    print("\n" + "=" * 80)
    print("EXAMPLE 9: Default Provider")
    print("=" * 80)
    await example_default_provider()


if __name__ == "__main__":
    asyncio.run(main())
