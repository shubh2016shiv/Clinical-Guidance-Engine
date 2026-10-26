# LLM Provider Adapter

Unified interface to language-model APIs. Currently supports OpenAI. Extensible to add Azure, Anthropic, or other providers.

## What This Module Does

The LLM Provider adapter lets the healthcare assistant talk to LLM APIs through a common interface. Instead of each agent writing its own API code, they use `LLMProvider` methods and get consistent responses, configuration, error handling, and token budgeting.

**Who uses this:**
- **Drug agent** — Chat Completions to write drug recommendations
- **Clinical guideline agent** — Responses API with file search for citations
- **Drug search** — Embeddings to turn text into vectors for Milvus

## Architecture

The adapter follows the **adapter pattern**: one interface, pluggable implementations.

```
┌─────────────────────────────────────┐
│ Drug Agent, Guideline Agent, Search │
└──────────────┬──────────────────────┘
               │ calls LLMProvider methods
               │
        ┌──────▼──────────────────┐
        │   LLMProvider (ABC)     │
        │  - execute()            │
        │  - execute_chat_completion()
        │  - token management     │
        └──────┬──────────────────┘
               │ inherited by
               │
        ┌──────▼──────────────────┐
        │   OpenAIProvider        │
        │  (openai/provider.py)   │
        └──────┬──────────────────┘
               │ calls SDK
               │
        ┌──────▼──────────────────┐
        │    OpenAI API           │
        │ (Chat, Responses, ...)  │
        └─────────────────────────┘
```

## Core Abstractions

### LLMConfig
Settings for any provider. Required at initialization.

```python
from src.adapters.llm_provider import LLMConfig, APIType

config = LLMConfig(
    model_name="gpt-4o",
    api_type=APIType.CHAT_COMPLETION,  # or APIType.RESPONSES
    token_limit=8192,  # hard limit for this request
    temperature=0.7,
    max_tokens=500,  # reserve tokens for response
)
```

| Field | Purpose |
| --- | --- |
| `model_name` | Which model to use (gpt-4o, gpt-4-turbo, etc.) |
| `api_type` | Chat Completions or Responses API |
| `token_limit` | Hard ceiling for input + output |
| `temperature` | 0–2. Lower = deterministic, higher = creative |
| `max_tokens` | Reserved for response; subtracts from available input tokens. Sent to OpenAI as `max_completion_tokens`, because the SDK deprecates `max_tokens` and newer models reject it |
| `top_p` | Nucleus sampling; overrides temperature if set |
| `frequency_penalty`, `presence_penalty` | Discourage repetition |
| `stop` | String(s) that end generation |
| `stream_mode` | Whether plain-prompt calls stream by default |

### LLMResponse
Unified reply object. Callers read `.content` for the answer.

```python
response = await provider.execute_chat_completion([
    {"role": "user", "content": "What is metformin?"}
])

print(response.content)  # The answer
print(response.model)  # gpt-4o
print(response.prompt_tokens)  # 12
print(response.completion_tokens)  # 45
```

### LLMProvider
Abstract base class. Subclasses implement:
- `execute()` — plain prompt to any API
- `execute_chat_completion()` — messages to Chat Completions
- Token validation and estimation

## Quick Start

### Chat Completions

For structured message exchanges:

```python
import asyncio
from src.adapters.llm_provider import LLMConfig, APIType, OpenAIProvider

async def main() -> None:
    config = LLMConfig(model_name="gpt-4o")
    provider = OpenAIProvider(config)
    try:
        response = await provider.execute_chat_completion([
            {"role": "system", "content": "You are a drug expert."},
            {"role": "user", "content": "What are SSRIs?"},
        ])
        print(response.content)
    finally:
        await provider.close()

asyncio.run(main())
```

### Responses API

For file search and citations:

```python
config = LLMConfig(model_name="gpt-4o", api_type=APIType.RESPONSES)
provider = OpenAIProvider(config)
try:
    response = await provider.execute_responses_api(
        "Summarize the clinical evidence for metformin.",
        stream=False,
    )
    print(response.output_text)  # OpenAI SDK result
finally:
    await provider.close()
```

### Streaming

Receive text as it arrives instead of waiting for completion:

```python
text_stream = await provider.execute_chat_completion(
    [{"role": "user", "content": "Say hello."}],
    stream=True,
)
async for piece in text_stream:
    print(piece, end="", flush=True)
```

### Embeddings

Turn text into vectors for Milvus:

```python
from src.adapters.llm_provider.openai import OpenAITextEmbeddingClient

client = OpenAITextEmbeddingClient(
    embedding_model_name="text-embedding-3-small",
    dimensions=768,
)
vector = await client.create_text_embedding("metformin")
# vector is a list of 768 floats for Milvus search
```

## Configuration

### Set API Key

The provider reads `OPENAI_API_KEY` from the environment or `.env` file. Do not hardcode keys.

```powershell
# Windows PowerShell
$env:OPENAI_API_KEY = "sk-..."
```

Or pass explicitly at creation:

```python
provider = OpenAIProvider(config, api_key="sk-...")
```

### From App Settings

The app reads config from environment variables and passes to providers:

```python
from src.config import get_settings

settings = get_settings()  # Reads OPENAI_* env vars
config = LLMConfig(
    model_name=settings.openai_model_name,
    api_type=settings.openai_api_type,
)
provider = OpenAIProvider(config)
```

## Error Handling

Every LLM error inherits from `LLMProviderError`. Catch it generically or handle specific cases.

```python
from src.adapters.llm_provider.exceptions import (
    LLMProviderError,
    LLMProviderTokenLimitError,
    LLMProviderAuthenticationError,
    LLMProviderRateLimitError,
)

try:
    response = await provider.execute_chat_completion(messages)
except LLMProviderTokenLimitError:
    # Truncate input and retry, or use a larger model
    print("Request too large for token budget")
except LLMProviderAuthenticationError:
    # API key is missing or invalid
    print("Check OPENAI_API_KEY")
except LLMProviderRateLimitError:
    # Implement backoff and retry
    print("Rate limited; waiting before retry")
except LLMProviderError as e:
    # Catch all LLM failures
    print(f"LLM request failed: {e}")
```

| Exception | When Raised | Your Action |
| --- | --- | --- |
| `LLMProviderTokenLimitError` | Pre-flight check: input + reserved output exceeds `token_limit` | Truncate or use larger model |
| `LLMProviderInitializationError` | No API key supplied (provider or embedding client construction) | Check `OPENAI_API_KEY` / settings |
| `LLMProviderAuthenticationError` | OpenAI rejects the API key (`openai.AuthenticationError`) | Check credentials and environment |
| `LLMProviderRateLimitError` | OpenAI returns 429 (`openai.RateLimitError`) | Implement exponential backoff |
| `LLMProviderExecutionError` | Any other SDK/request failure (4xx, 5xx, timeout, etc.) | Retry with backoff or fail gracefully |
| `LLMProviderStreamingError` | A stream that already started fails mid-iteration | Log and fall back to non-streaming |

All three API paths — Chat Completions, Responses, and Embeddings — route SDK exceptions through the same mapping in [openai/error_mapping.py](openai/error_mapping.py), so `except LLMProviderError` reliably catches every LLM failure regardless of which API raised it.

`LLMProviderUnsupportedOperationError` is defined but not currently raised; it's reserved for a future provider that can't support an operation another provider offers (e.g., no streaming).

## Token Management

The provider checks whether a request fits before sending it to OpenAI.

### Token Estimation

```python
tokens = LLMProvider.estimate_tokens("What is metformin?")
# Estimates using tiktoken; falls back to text length ÷ 4
```

### Validation

```python
if provider.validate_token_limit(prompt="...", system_prompt="..."):
    # Safe to send
    response = await provider.execute(prompt, system_prompt)
else:
    # Token limit would be exceeded
    print("Request too large")
```

The provider reserves `max_tokens` for the response, so:
```
total = estimate(prompt) + estimate(system_prompt) + max_tokens
if total > token_limit:
    raise LLMProviderTokenLimitError
```

## Adding a New Provider

To support Azure, Anthropic, or another service:

1. **Create a folder:**
   ```
   src/adapters/llm_provider/<provider_name>/
   ```

2. **Inherit from `LLMProvider`:**
   ```python
   from src.adapters.llm_provider.base import LLMProvider, LLMConfig

   class AnthropicProvider(LLMProvider):
       async def execute(self, prompt, system_prompt=None, **kwargs):
           # Call Anthropic SDK
           pass

       async def execute_chat_completion(self, messages, **kwargs):
           # Call Anthropic SDK
           pass
   ```

3. **Implement required methods:**
   - `execute()` — plain text to any API
   - `execute_chat_completion()` — messages to Chat Completions

4. **Optional:** Add provider-specific operations (file search, embeddings).

5. **Export from `__init__.py`:**
   ```python
   from src.adapters.llm_provider.anthropic.provider import AnthropicProvider
   __all__ = ["AnthropicProvider", ...]
   ```

See [openai/provider.py](openai/provider.py) for a complete example.

## File Reference

| File | Purpose |
| --- | --- |
| [base.py](base.py) | `LLMProvider`, `LLMConfig`, `LLMResponse`, `APIType`, `StreamMode` |
| [exceptions.py](exceptions.py) | Error types: authentication, rate limit, token limit, streaming, etc. |
| [schema.py](schema.py) | TypedDict for request shapes (ChatCompletionRequest, ResponsesApiRequest) |
| [openai/](openai/) | OpenAI implementation. See [openai/README.md](openai/README.md) |
| [openai/provider.py](openai/provider.py) | `OpenAIProvider` — main entry point |
| [openai/chat_completions.py](openai/chat_completions.py) | Chat Completions client with token validation |
| [openai/responses_api.py](openai/responses_api.py) | Responses API client (file search, citations) |
| [openai/embeddings.py](openai/embeddings.py) | `OpenAITextEmbeddingClient` for vector creation |
| [openai/error_mapping.py](openai/error_mapping.py) | Maps OpenAI SDK exceptions to the `LLMProviderError` hierarchy |
| [__init__.py](__init__.py) | Public exports (`OpenAIProvider`, `LLMConfig`, `APIType`, `exceptions`) |

## Glossary

- **LLM:** Large language model that generates text.
- **Provider:** Subclass of `LLMProvider` that talks to a specific API (OpenAI, Azure, Anthropic).
- **API Type:** Chat Completions (messages) or Responses API (file search, tool use).
- **Token:** Piece of text as counted by the model. Affects cost and speed.
- **Stream:** Receive response text as it's generated, instead of waiting for completion.
- **Embedding:** Vector (list of numbers) representing text. Used for similarity search.
