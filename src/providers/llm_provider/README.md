# LLM Provider System

A professional, extensible, and maintainable system for managing Large Language Model providers in your Python application.

## 🌟 Features

- **Multiple Provider Support**: OpenAI, with easy extension for Anthropic, Gemini, etc.
- **Dual API Support**: 
  - Chat Completions API (standard chat interface)
  - Responses API (OpenAI's conversational interface)
- **Streaming Support**: Full support for real-time streaming responses
- **Type-Safe Configuration**: Using Python dataclasses and enums
- **Centralized Management**: Single point of configuration and initialization
- **Flexible Configuration**: Environment, code, or runtime configuration
- **Comprehensive Error Handling**: Specific exceptions for different failure modes
- **Token Management**: Built-in token estimation and validation
- **Production Ready**: Logging, retries, timeouts, and error handling

## 📦 Installation

```bash
# Install required dependencies
pip install openai python-dotenv pydantic
```

## 🚀 Quick Start

### Basic Usage

```python
from src.providers import create_llm_provider

# Create provider with defaults
provider = create_llm_provider(
    provider_type="openai",
    model_name="gpt-4"
)

# Execute a simple query
response = await provider.execute(
    prompt="What is the capital of France?",
    system_prompt="You are a helpful assistant."
)

print(response.content)
```

### Streaming Responses

```python
from src.providers import create_llm_provider, StreamMode

# Enable streaming
provider = create_llm_provider(
    provider_type="openai",
    model_name="gpt-4",
    stream_mode=StreamMode.ENABLED
)

# Stream the response
stream = await provider.execute(
    prompt="Write a story about coding"
)

async for chunk in stream:
    print(chunk, end="", flush=True)
```

### Using Responses API

```python
from src.providers import create_llm_provider, APIType

# Configure for Responses API
provider = create_llm_provider(
    provider_type="openai",
    model_name="gpt-4",
    api_type=APIType.RESPONSES
)

# First message
response = await provider.execute_responses_api(
    input_message="Hello! I need help with Python.",
    instructions="You are a helpful Python programming assistant."
)

# Continue conversation with context
response2 = await provider.execute_responses_api(
    input_message="What are decorators?",
    instructions="You are a helpful Python programming assistant.",
    previous_response_id=response.response_id
)
```

### Advanced Configuration

```python
from src.providers import create_llm_provider, LLMConfig, APIType, StreamMode

# Create detailed configuration
config = LLMConfig(
    model_name="gpt-4-turbo",
    api_type=APIType.RESPONSES,
    token_limit=128000,
    temperature=0.8,
    max_tokens=2000,
    top_p=0.9,
    frequency_penalty=0.5,
    presence_penalty=0.3,
    stream_mode=StreamMode.ENABLED,
    timeout=120,
    max_retries=3
)

# Create provider with config
provider = create_llm_provider(
    provider_type="openai",
    config=config
)
```

## 📁 Project Structure

```
src/providers/
├── __init__.py              # Public API exports
├── base.py                  # Abstract base classes and interfaces
├── factory.py               # Provider factory and registry
├── openai_provider.py       # OpenAI implementation
├── exceptions.py            # Custom exception classes
└── README.md               # This file

Optional:
├── gemini_provider.py       # Google Gemini implementation
├── anthropic_provider.py    # Anthropic Claude implementation
└── usage_examples.py        # Comprehensive usage examples
```

## 🔧 Configuration

### Environment Variables

```bash
# .env file
OPENAI_API_KEY=your_api_key_here
DEFAULT_LLM_PROVIDER=openai
DEFAULT_MODEL_NAME=gpt-4
```

### Application Settings (config.py)

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    openai_api_key: str
    default_llm_provider: str = "openai"
    default_model_name: str = "gpt-4"
    
    class Config:
        env_file = ".env"
```

## 🎯 API Types

### Chat Completion API
Standard chat interface with message history:

```python
provider = create_llm_provider(
    provider_type="openai",
    api_type=APIType.CHAT_COMPLETION
)

messages = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Hello!"}
]

response = await provider.execute_chat_completion(messages=messages)
```

### Responses API
Simplified conversational interface:

```python
provider = create_llm_provider(
    provider_type="openai",
    api_type=APIType.RESPONSES
)

response = await provider.execute_responses_api(
    input_message="Hello!",
    instructions="You are helpful.",
    previous_response_id=None  # For conversation continuity
)
```

## 🌊 Streaming Modes

- **StreamMode.DISABLED**: Traditional request/response
- **StreamMode.ENABLED**: Real-time streaming
- **StreamMode.AUTO**: Provider decides based on context

```python
# Non-streaming
provider = create_llm_provider(stream_mode=StreamMode.DISABLED)
response = await provider.execute(prompt="Hello")
print(response.content)

# Streaming
provider = create_llm_provider(stream_mode=StreamMode.ENABLED)
stream = await provider.execute(prompt="Hello")
async for chunk in stream:
    print(chunk, end="")
```

## 🛡️ Error Handling

```python
from src.providers import (
    create_llm_provider,
    LLMProviderError,
    LLMProviderTokenLimitError,
    LLMProviderRateLimitError
)

try:
    provider = create_llm_provider(provider_type="openai")
    response = await provider.execute(prompt="Your prompt")
    
except LLMProviderTokenLimitError as e:
    print(f"Token limit exceeded: {e}")
    # Handle token limit - maybe truncate prompt
    
except LLMProviderRateLimitError as e:
    print(f"Rate limited: {e}")
    # Implement retry with backoff
    
except LLMProviderError as e:
    print(f"Provider error: {e}")
    # General provider error handling
```

## 🔌 Extending with New Providers

### 1. Create Provider Implementation

```python
# src/providers/custom_provider.py
from .base import LLMProvider, LLMConfig, LLMResponse

class CustomProvider(LLMProvider):
    def __init__(self, config: LLMConfig, api_key: str = None, **kwargs):
        super().__init__(config)
        self.api_key = api_key
        # Initialize your provider client
    
    async def execute(self, prompt: str, **kwargs):
        # Implement execution logic
        pass
    
    async def execute_chat_completion(self, messages, **kwargs):
        # Implement chat completion
        pass
    
    async def execute_responses_api(self, input_message, **kwargs):
        # Implement if applicable
        raise LLMProviderUnsupportedOperationError(
            "Responses API not supported"
        )
```

### 2. Register Provider

```python
from src.providers import register_provider
from .custom_provider import CustomProvider

register_provider("custom", CustomProvider)
```

### 3. Use New Provider

```python
provider = create_llm_provider(
    provider_type="custom",
    model_name="custom-model-v1"
)
```

## 📊 Response Object

All providers return a standardized `LLMResponse` object:

```python
@dataclass
class LLMResponse:
    content: str                    # Generated text
    model: str                      # Model used
    provider_type: str             # Provider identifier
    api_type: APIType              # API type used
    prompt_tokens: Optional[int]   # Input tokens
    completion_tokens: Optional[int]  # Output tokens
    total_tokens: Optional[int]    # Total tokens
    finish_reason: Optional[str]   # Why generation stopped
    response_id: Optional[str]     # Unique response ID
    metadata: Dict[str, Any]       # Additional data
```

## 🎛️ Configuration Options

### LLMConfig Parameters

```python
@dataclass
class LLMConfig:
    # Model settings
    model_name: str                    # Required
    api_type: APIType                  # CHAT_COMPLETION or RESPONSES
    token_limit: int = 8192           # Max context window
    
    # Generation parameters
    temperature: float = 0.7           # 0.0-1.0 (creativity)
    max_tokens: Optional[int] = None   # Max output tokens
    top_p: Optional[float] = None      # 0.0-1.0 (nucleus sampling)
    frequency_penalty: Optional[float] = None  # 0.0-2.0
    presence_penalty: Optional[float] = None   # 0.0-2.0
    stop: Optional[Union[str, List[str]]] = None  # Stop sequences
    
    # Streaming
    stream_mode: StreamMode = StreamMode.DISABLED
    
    # Reliability
    timeout: int = 60                  # Request timeout (seconds)
    max_retries: int = 3              # Retry attempts
    retry_delay: float = 1.0          # Initial retry delay (seconds)
    
    # Additional settings
    extra_params: Dict[str, Any] = field(default_factory=dict)
```

## 🔄 Dynamic Configuration Updates

```python
# Create provider
provider = create_llm_provider(
    provider_type="openai",
    temperature=0.7
)

# Update settings at runtime
provider.update_config(
    temperature=0.9,
    max_tokens=1000
)

# New executions use updated config
response = await provider.execute(prompt="Generate creative text")
```

## 💾 Provider Caching

For improved performance when reusing the same configuration:

```python
from src.providers import get_cached_provider, clear_provider_cache

# Get or create cached provider
provider = get_cached_provider(
    provider_type="openai",
    model_name="gpt-4",
    api_type="chat_completion",
    stream_mode="disabled"
)

# Clear cache when needed (e.g., after config changes)
clear_provider_cache()
```

## 🧪 Testing

### Unit Test Example

```python
import pytest
from src.providers import create_llm_provider, APIType, StreamMode

@pytest.mark.asyncio
async def test_basic_execution():
    provider = create_llm_provider(
        provider_type="openai",
        model_name="gpt-4"
    )
    
    response = await provider.execute(
        prompt="Say 'test successful'"
    )
    
    assert response.content is not None
    assert response.model == "gpt-4"
    assert response.total_tokens > 0

@pytest.mark.asyncio
async def test_streaming():
    provider = create_llm_provider(
        provider_type="openai",
        stream_mode=StreamMode.ENABLED
    )
    
    stream = await provider.execute(prompt="Count to 5")
    chunks = []
    
    async for chunk in stream:
        chunks.append(chunk)
    
    assert len(chunks) > 0
    assert ''.join(chunks)  # Non-empty result
```

## 📝 Best Practices

### 1. Use Configuration Objects for Complex Settings

```python
# ✅ Good - Reusable configuration
config = LLMConfig(
    model_name="gpt-4",
    temperature=0.8,
    max_tokens=2000
)
provider = create_llm_provider("openai", config=config)

# ❌ Avoid - Repetitive parameter passing
provider = create_llm_provider(
    "openai",
    model_name="gpt-4",
    temperature=0.8,
    max_tokens=2000
)
```

### 2. Always Handle Exceptions

```python
# ✅ Good - Proper error handling
try:
    response = await provider.execute(prompt="Your prompt")
except LLMProviderError as e:
    logger.error(f"LLM error: {e}")
    # Implement fallback logic

# ❌ Avoid - Unhandled errors
response = await provider.execute(prompt="Your prompt")
```

### 3. Validate Token Limits Before Expensive Calls

```python
# ✅ Good - Check before calling
if provider.validate_token_limit(prompt, max_tokens=1000):
    response = await provider.execute(prompt=prompt)
else:
    prompt = truncate_prompt(prompt)
    response = await provider.execute(prompt=prompt)
```

### 4. Use Streaming for Long Responses

```python
# ✅ Good - Stream for better UX
provider = create_llm_provider(stream_mode=StreamMode.ENABLED)
stream = await provider.execute(prompt="Write a long article")

async for chunk in stream:
    display_to_user(chunk)  # Show immediately

# ❌ Avoid - User waits for complete response
provider = create_llm_provider(stream_mode=StreamMode.DISABLED)
response = await provider.execute(prompt="Write a long article")
display_to_user(response.content)  # Long wait
```

### 5. Reuse Providers When Possible

```python
# ✅ Good - Reuse provider instance
provider = create_llm_provider("openai")
for prompt in prompts:
    response = await provider.execute(prompt=prompt)

# ❌ Avoid - Creating new providers repeatedly
for prompt in prompts:
    provider = create_llm_provider("openai")
    response = await provider.execute(prompt=prompt)
```

## 🔍 Monitoring and Debugging

### Getting Provider Information

```python
provider = create_llm_provider("openai", model_name="gpt-4")

# Get comprehensive info
info = provider.get_model_info()
print(f"Provider: {info['provider_type']}")
print(f"Model: {info['model_name']}")
print(f"API Type: {info['api_type']}")
print(f"Token Limit: {info['token_limit']}")
```

### Token Estimation

```python
# Estimate tokens before calling
prompt = "Your long prompt here..."
estimated_tokens = provider.estimate_tokens(prompt)
print(f"Estimated tokens: {estimated_tokens}")

# Validate against limits
is_valid = provider.validate_token_limit(
    prompt=prompt,
    max_tokens=1000
)
```

### Response Analysis

```python
response = await provider.execute(prompt="Test")

# Analyze usage
print(f"Prompt tokens: {response.prompt_tokens}")
print(f"Completion tokens: {response.completion_tokens}")
print(f"Total tokens: {response.total_tokens}")
print(f"Finish reason: {response.finish_reason}")

# Convert to dict for logging
response_dict = response.to_dict()
logger.info(f"Response data: {response_dict}")
```

## 🚀 Advanced Usage

### Multi-turn Conversations

```python
# Using Chat Completions
provider = create_llm_provider(api_type=APIType.CHAT_COMPLETION)

messages = [
    {"role": "system", "content": "You are a helpful assistant."}
]

# Turn 1
messages.append({"role": "user", "content": "Hello!"})
response1 = await provider.execute_chat_completion(messages=messages)
messages.append({"role": "assistant", "content": response1.content})

# Turn 2
messages.append({"role": "user", "content": "Tell me a joke."})
response2 = await provider.execute_chat_completion(messages=messages)
```

### Using Tools/Function Calling

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"]
            }
        }
    }
]

provider = create_llm_provider(api_type=APIType.RESPONSES)
response = await provider.execute_responses_api(
    input_message="What's the weather in Paris?",
    instructions="You have access to weather data.",
    tools=tools
)
```

### Parallel Requests

```python
import asyncio

provider = create_llm_provider("openai")

prompts = [
    "Summarize quantum physics",
    "Explain machine learning",
    "Describe blockchain"
]

# Execute in parallel
responses = await asyncio.gather(*[
    provider.execute(prompt=p) for p in prompts
])

for response in responses:
    print(response.content)
```

## 📚 Additional Resources

- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Usage Examples](./usage_examples.py) - Comprehensive code examples
- [Exception Reference](./exceptions.py) - All exception types

## 🤝 Contributing

To add a new provider:

1. Create `your_provider.py` extending `LLMProvider`
2. Implement required abstract methods
3. Register in `factory.py`
4. Add tests
5. Update documentation

## 📄 License

[Your License Here]

## 🆘 Support

For issues and questions:
- Check [usage_examples.py](./usage_examples.py) for common patterns
- Review exception types in [exceptions.py](./exceptions.py)
- Enable debug logging: `logger.setLevel(logging.DEBUG)`

---

**Version**: 2.0.0  
**Last Updated**: 2025