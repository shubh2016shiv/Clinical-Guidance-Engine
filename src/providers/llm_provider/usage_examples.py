"""
LLM Provider Usage Examples

This file demonstrates various usage patterns for the LLM provider system.
Copy these examples into your application code as needed.
"""

import asyncio
from typing import AsyncIterator

from src.providers.llm_provider.factory import create_llm_provider, get_default_provider
from src.providers.llm_provider.base import LLMConfig, APIType, StreamMode, LLMResponse
from src.providers.exceptions import LLMProviderError


# ============================================================================
# EXAMPLE 1: Basic Usage - Simple Chat Completion
# ============================================================================


async def example_basic_chat():
    """Basic non-streaming chat completion"""

    # Create provider with defaults
    provider = create_llm_provider(provider_type="openai", model_name="gpt-4")

    # Execute simple prompt
    response: LLMResponse = await provider.execute(
        prompt="What is the capital of France?",
        system_prompt="You are a helpful geography assistant.",
    )

    print(f"Response: {response.content}")
    print(f"Tokens used: {response.total_tokens}")
    print(f"Model: {response.model}")


# ============================================================================
# EXAMPLE 2: Streaming Chat Completion
# ============================================================================


async def example_streaming_chat():
    """Streaming chat completion with real-time output"""

    # Create provider with streaming enabled
    provider = create_llm_provider(
        provider_type="openai", model_name="gpt-4", stream_mode=StreamMode.ENABLED
    )

    # Execute with streaming
    stream: AsyncIterator[str] = await provider.execute(
        prompt="Write a short poem about coding",
        system_prompt="You are a creative poet.",
    )

    # Process stream chunks as they arrive
    print("Streaming response:")
    async for chunk in stream:
        print(chunk, end="", flush=True)
    print()  # New line after streaming completes


# ============================================================================
# EXAMPLE 3: Responses API (Non-Streaming)
# ============================================================================


async def example_responses_api():
    """Using OpenAI Responses API for conversational context"""

    # Create provider configured for Responses API
    provider = create_llm_provider(
        provider_type="openai",
        model_name="gpt-4",
        api_type=APIType.RESPONSES,
        stream_mode=StreamMode.DISABLED,
    )

    # First message in conversation
    response: LLMResponse = await provider.execute_responses_api(
        input_message="Hello! I'm working on a Python project.",
        instructions="You are a helpful Python programming assistant. Be concise and practical.",
    )

    print(f"Assistant: {response.content}")

    # Continue conversation with context
    response2: LLMResponse = await provider.execute_responses_api(
        input_message="Can you suggest some best practices?",
        instructions="You are a helpful Python programming assistant. Be concise and practical.",
        previous_response_id=response.response_id,  # Maintains context
    )

    print(f"Assistant: {response2.content}")


# ============================================================================
# EXAMPLE 4: Responses API (Streaming)
# ============================================================================


async def example_responses_api_streaming():
    """Streaming with Responses API"""

    # Create provider with streaming enabled
    provider = create_llm_provider(
        provider_type="openai",
        model_name="gpt-4",
        api_type=APIType.RESPONSES,
        stream_mode=StreamMode.ENABLED,
    )

    # Execute with streaming
    stream: AsyncIterator[str] = await provider.execute_responses_api(
        input_message="Explain async programming in Python",
        instructions="You are an expert Python developer. Explain concepts clearly.",
        stream=True,
    )

    # Process stream
    print("Streaming response:")
    full_response = []
    async for chunk in stream:
        print(chunk, end="", flush=True)
        full_response.append(chunk)

    print(f"\n\nTotal length: {len(''.join(full_response))} characters")


# ============================================================================
# EXAMPLE 5: Advanced Configuration with LLMConfig
# ============================================================================


async def example_advanced_config():
    """Using LLMConfig for detailed configuration"""

    # Create detailed configuration
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

    # Create provider with config
    provider = create_llm_provider(provider_type="openai", config=config)

    # Execute
    response: LLMResponse = await provider.execute(
        prompt="Write a creative story about AI"
    )

    print(f"Response: {response.content}")
    print(f"Configuration used: {provider.config.to_dict()}")


# ============================================================================
# EXAMPLE 6: Dynamic Configuration Updates
# ============================================================================


async def example_dynamic_config():
    """Updating provider configuration at runtime"""

    # Create provider
    provider = create_llm_provider(
        provider_type="openai", model_name="gpt-4", temperature=0.7
    )

    # First call with default temperature
    response1: LLMResponse = await provider.execute(
        prompt="Generate a random number between 1 and 10"
    )
    print(f"Response 1: {response1.content}")

    # Update configuration dynamically
    provider.update_config(temperature=1.0, max_tokens=50)

    # Second call with updated temperature
    response2: LLMResponse = await provider.execute(
        prompt="Generate a random number between 1 and 10"
    )
    print(f"Response 2: {response2.content}")


# ============================================================================
# EXAMPLE 7: Using Tools with Responses API
# ============================================================================


async def example_with_tools():
    """Using function calling/tools with Responses API"""

    # Define tools
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"}
                    },
                    "required": ["location"],
                },
            },
        }
    ]

    # Create provider
    provider = create_llm_provider(
        provider_type="openai", model_name="gpt-4", api_type=APIType.RESPONSES
    )

    # Execute with tools
    response: LLMResponse = await provider.execute_responses_api(
        input_message="What's the weather in Paris?",
        instructions="You are a helpful assistant with access to weather data.",
        tools=tools,
    )

    print(f"Response: {response.content}")
    print(f"Metadata: {response.metadata}")


# ============================================================================
# EXAMPLE 8: Error Handling
# ============================================================================


async def example_error_handling():
    """Proper error handling for LLM operations"""

    try:
        # Create provider
        provider = create_llm_provider(provider_type="openai", model_name="gpt-4")

        # Attempt execution with very long prompt
        long_prompt = "A" * 200000  # Exceeds token limit

        try:
            response: LLMResponse = await provider.execute(prompt=long_prompt)
            print(f"Unexpected success: {response.content[:100]}...")
        except LLMProviderError:
            # This is expected - token limit should be exceeded
            pass

    except LLMProviderError as e:
        print(f"Provider error occurred: {e}")
        # Handle specific provider errors
        # Could implement retry logic, fallback models, etc.

    except Exception as e:
        print(f"Unexpected error: {e}")


# ============================================================================
# EXAMPLE 9: Multi-turn Conversation with Chat Completions
# ============================================================================


async def example_multi_turn_conversation():
    """Managing multi-turn conversations with Chat Completions API"""

    provider = create_llm_provider(provider_type="openai", model_name="gpt-4")

    # Maintain conversation history
    messages = [
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": "What is a closure in Python?"},
    ]

    # First turn
    response1: LLMResponse = await provider.execute_chat_completion(messages=messages)
    print(f"Assistant: {response1.content}")

    # Add to history
    messages.append({"role": "assistant", "content": response1.content})
    messages.append({"role": "user", "content": "Can you give me an example?"})

    # Second turn
    response2: LLMResponse = await provider.execute_chat_completion(messages=messages)
    print(f"Assistant: {response2.content}")


# ============================================================================
# EXAMPLE 10: Provider Information and Monitoring
# ============================================================================


async def example_provider_info():
    """Getting information about provider configuration"""

    provider = create_llm_provider(
        provider_type="openai",
        model_name="gpt-4",
        api_type=APIType.RESPONSES,
        stream_mode=StreamMode.ENABLED,
    )

    # Get provider information
    info = provider.get_model_info()
    print("Provider Information:")
    print(f"  Type: {info['provider_type']}")
    print(f"  Model: {info['model_name']}")
    print(f"  API: {info['api_type']}")
    print(f"  Token Limit: {info['token_limit']}")
    print(f"  Streaming: {info['stream_mode']}")
    print(f"  Configuration: {info['config']}")

    # Validate token limits
    test_prompt = "This is a test prompt"
    is_valid = provider.validate_token_limit(test_prompt, max_tokens=1000)
    print(f"\nToken validation passed: {is_valid}")


# ============================================================================
# EXAMPLE 11: Using Default Provider
# ============================================================================


async def example_default_provider():
    """Using the default provider from application settings"""

    # Get default provider (uses settings from config)
    provider = get_default_provider()

    # Execute with defaults
    response: LLMResponse = await provider.execute(prompt="What is machine learning?")

    print(f"Response: {response.content}")


# ============================================================================
# Main execution
# ============================================================================


async def main():
    """Run all examples"""

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
    print("EXAMPLE 4: Streaming Responses API")
    print("=" * 80)
    await example_responses_api_streaming()

    print("\n" + "=" * 80)
    print("EXAMPLE 10: Provider Information")
    print("=" * 80)
    await example_provider_info()

    # Add more example calls as needed


if __name__ == "__main__":
    # Run examples
    asyncio.run(main())
