# Embedding Provider System

A professional, extensible, and maintainable system for managing embedding generation providers in your Python application.

## 🌟 Features

- **Multiple Provider Support**: OpenAI, Gemini, with easy extension for Cohere, HuggingFace, etc.
- **Task-Specific Embeddings**: Optimize for similarity, classification, clustering, or retrieval
- **Batch Processing**: Efficient batch embedding generation with automatic chunking
- **Sync & Async Support**: Both synchronous and asynchronous APIs
- **Type-Safe Configuration**: Using Python dataclasses and enums
- **Centralized Management**: Single point of configuration and initialization
- **Comprehensive Error Handling**: Specific exceptions for different failure modes
- **Text Preprocessing**: Built-in text validation and preprocessing
- **Production Ready**: Logging, retries, timeouts, and error handling

## 📦 Installation

```bash
# Install required dependencies
pip install openai google-generativeai numpy python-dotenv pydantic
```

## 🚀 Quick Start

### Basic Embedding Generation

```python
from src.providers.embedding_provider import create_embedding_provider

# Create provider
provider = create_embedding_provider(
    provider_type="openai",
    model_name="text-embedding-3-small"
)

# Generate single embedding (async)
embedding = await provider.generate_embedding("Your text here")
print(f"Dimension: {len(embedding)}")

# Generate single embedding (sync)
embedding = provider.generate_embedding_sync("Your text here")
```

### Batch Embedding Generation

```python
from src.providers.embedding_provider import create_embedding_provider

# Create provider
provider = create_embedding_provider(
    provider_type="openai",
    model_name="text-embedding-3-small"
)

# Generate batch embeddings
texts = [
    "First document",
    "Second document",
    "Third document"
]

embeddings = await provider.generate_embeddings_batch(texts)
print(f"Generated {len(embeddings)} embeddings")
```

### Task-Specific Embeddings

```python
from src.providers.embedding_provider import (
    create_embedding_provider,
    EmbeddingTaskType
)

# For semantic similarity
provider = create_embedding_provider(
    provider_type="gemini",
    model_name="text-embedding-004",
    task_type=EmbeddingTaskType.SEMANTIC_SIMILARITY
)

# For retrieval queries
query_provider = create_embedding_provider(
    provider_type="gemini",
    task_type=EmbeddingTaskType.RETRIEVAL_QUERY
)

# For retrieval documents
doc_provider = create_embedding_provider(
    provider_type="gemini",
    task_type=EmbeddingTaskType.RETRIEVAL_DOCUMENT
)
```

## 📁 Project Structure

```
src/providers/embedding_provider/
├── __init__.py                      # Public API exports
├── base.py                          # Abstract base classes and interfaces
├── factory.py                       # Provider factory and registry
├── openai_embedding_provider.py    # OpenAI implementation
├── gemini_embedding_provider.py    # Gemini implementation
├── embedding_exceptions.py          # Custom exception classes
├── embedding_usage_examples.py      # Comprehensive usage examples
└── EMBEDDING_README.md             # This file
```

## 🔧 Configuration

### Environment Variables

```bash
# .env file
OPENAI_API_KEY=your_openai_key
GEMINI_API_KEY=your_gemini_key
DEFAULT_EMBEDDING_PROVIDER=openai
DEFAULT_EMBEDDING_MODEL=text-embedding-3-small
```

### Application Settings

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    openai_api_key: str
    gemini_api_key: str
    default_embedding_provider: str = "openai"
    default_embedding_model: str = "text-embedding-3-small"
    
    class Config:
        env_file = ".env"
```

## 🎯 Task Types

### Available Task Types

- **SEMANTIC_SIMILARITY**: General similarity comparisons (default)
- **CLASSIFICATION**: Text classification tasks
- **CLUSTERING**: Grouping similar documents
- **RETRIEVAL_QUERY**: Query embeddings in search systems
- **RETRIEVAL_DOCUMENT**: Document embeddings in search systems

```python
from src.providers.embedding_provider import EmbeddingTaskType

# Use appropriate task type for your use case
provider = create_embedding_provider(
    provider_type="gemini",
    task_type=EmbeddingTaskType.RETRIEVAL_DOCUMENT
)
```

## 🏭 Supported Providers

### OpenAI

**Models:**
- `text-embedding-3-small` (1536 dimensions) - Cost-effective
- `text-embedding-3-large` (3072 dimensions) - Higher quality
- `text-embedding-ada-002` (1536 dimensions) - Legacy

```python
provider = create_embedding_provider(
    provider_type="openai",
    model_name="text-embedding-3-small"
)
```

### Gemini (Google)

**Models:**
- `text-embedding-004` (768 dimensions) - Latest model
- `embedding-001` (768 dimensions) - Legacy

```python
provider = create_embedding_provider(
    provider_type="gemini",
    model_name="text-embedding-004",
    task_type=EmbeddingTaskType.SEMANTIC_SIMILARITY
)
```

## 🎛️ Advanced Configuration

### Using EmbeddingConfig

```python
from src.providers.embedding_provider import (
    create_embedding_provider,
    EmbeddingConfig,
    EmbeddingTaskType
)

# Create detailed configuration
config = EmbeddingConfig(
    model_name="text-embedding-3-large",
    embedding_dimension=3072,
    task_type=EmbeddingTaskType.RETRIEVAL_DOCUMENT,
    batch_size=50,
    timeout=120,
    max_retries=3,
    strip_whitespace=True,
    remove_empty=True,
    max_text_length=8000
)

# Create provider with config
provider = create_embedding_provider(
    provider_type="openai",
    config=config
)
```

### Configuration Parameters

```python
@dataclass
class EmbeddingConfig:
    # Model settings
    model_name: str                        # Required
    embedding_dimension: int = 1536        # Vector dimension
    
    # Task configuration
    task_type: EmbeddingTaskType = SEMANTIC_SIMILARITY
    
    # Batch processing
    batch_size: int = 100                  # Max texts per batch
    
    # Performance
    timeout: int = 60                      # Request timeout (seconds)
    max_retries: int = 3                   # Retry attempts
    retry_delay: float = 1.0               # Initial retry delay
    
    # Text preprocessing
    strip_whitespace: bool = True          # Remove whitespace
    remove_empty: bool = True              # Filter empty texts
    max_text_length: Optional[int] = None  # Max characters
    
    # Additional settings
    extra_params: Dict[str, Any] = {}      # Provider-specific
```

## 🔄 Dynamic Configuration Updates

```python
# Create provider
provider = create_embedding_provider(
    provider_type="openai",
    batch_size=10
)

# Update settings at runtime
provider.update_config(
    batch_size=100,
    max_text_length=5000
)
```

## 🛡️ Error Handling

```python
from src.providers.embedding_provider import (
    create_embedding_provider,
    EmbeddingProviderError,
    EmbeddingGenerationError,
    EmbeddingDimensionMismatchError
)

try:
    provider = create_embedding_provider(provider_type="openai")
    embedding = await provider.generate_embedding(text)
    
except EmbeddingDimensionMismatchError as e:
    print(f"Dimension mismatch: {e}")
    # Handle dimension issues
    
except EmbeddingGenerationError as e:
    print(f"Generation failed: {e}")
    print(f"Details: {e.details}")
    # Implement retry logic
    
except EmbeddingProviderError as e:
    print(f"Provider error: {e}")
    # General error handling
```

## 📊 Response Objects

### EmbeddingResponse

```python
@dataclass
class EmbeddingResponse:
    embeddings: List[List[float]]     # List of embedding vectors
    model: str                         # Model used
    provider_type: str                 # Provider identifier
    dimension: int                     # Embedding dimension
    total_tokens: Optional[int]        # Token usage
    metadata: Dict[str, Any]           # Additional data
```

## 🔌 Extending with New Providers

### 1. Create Provider Implementation

```python
# src/providers/embedding_provider/custom_provider.py
from .base import EmbeddingProvider, EmbeddingConfig

class CustomEmbeddingProvider(EmbeddingProvider):
    def __init__(self, config: EmbeddingConfig, api_key: str = None, **kwargs):
        super().__init__(config)
        self.api_key = api_key
        # Initialize your provider client
    
    async def generate_embedding(self, text: str, **kwargs):
        # Implement embedding generation
        pass
    
    def generate_embedding_sync(self, text: str, **kwargs):
        # Implement sync embedding generation
        pass
    
    async def generate_embeddings_batch(self, texts, **kwargs):
        # Implement batch generation
        pass
    
    def generate_embeddings_batch_sync(self, texts, **kwargs):
        # Implement sync batch generation
        pass
```

### 2. Register Provider

```python
from src.providers.embedding_provider import register_embedding_provider
from .custom_provider import CustomEmbeddingProvider

register_embedding_provider("custom", CustomEmbeddingProvider)
```

### 3. Use New Provider

```python
provider = create_embedding_provider(
    provider_type="custom",
    model_name="custom-model-v1"
)
```

## 💾 Provider Caching

```python
from src.providers.embedding_provider import (
    get_cached_embedding_provider,
    clear_embedding_provider_cache
)

# Get or create cached provider
provider = get_cached_embedding_provider(
    provider_type="openai",
    model_name="text-embedding-3-small",
    embedding_dimension=1536,
    task_type="semantic_similarity"
)

# Clear cache when needed
clear_embedding_provider_cache()
```

## 🧪 Testing

```python
import pytest
from src.providers.embedding_provider import create_embedding_provider

@pytest.mark.asyncio
async def test_basic_embedding():
    provider = create_embedding_provider(
        provider_type="openai",
        model_name="text-embedding-3-small"
    )
    
    embedding = await provider.generate_embedding("test text")
    
    assert len(embedding) == 1536
    assert all(isinstance(x, float) for x in embedding)

@pytest.mark.asyncio
async def test_batch_embeddings():
    provider = create_embedding_provider(provider_type="openai")
    
    texts = ["text 1", "text 2", "text 3"]
    embeddings = await provider.generate_embeddings_batch(texts)
    
    assert len(embeddings) == 3
    assert all(len(emb) == 1536 for emb in embeddings)
```

## 📝 Best Practices

### 1. Use Appropriate Task Types

```python
# ✅ Good - Use specific task types
query_provider = create_embedding_provider(
    task_type=EmbeddingTaskType.RETRIEVAL_QUERY
)
doc_provider = create_embedding_provider(
    task_type=EmbeddingTaskType.RETRIEVAL_DOCUMENT
)

# ❌ Avoid - Using generic type for specialized tasks
provider = create_embedding_provider()  # Defaults to SEMANTIC_SIMILARITY
```

### 2. Batch When Possible

```python
# ✅ Good - Batch processing
texts = ["text1", "text2", "text3"]
embeddings = await provider.generate_embeddings_batch(texts)

# ❌ Avoid - Individual calls in loop
embeddings = []
for text in texts:
    emb = await provider.generate_embedding(text)
    embeddings.append(emb)
```

### 3. Handle Errors Gracefully

```python
# ✅ Good - Proper error handling
try:
    embeddings = await provider.generate_embeddings_batch(texts)
except EmbeddingProviderError as e:
    logger.error(f"Embedding failed: {e}")
    # Implement fallback or retry
```

### 4. Reuse Provider Instances

```python
# ✅ Good - Reuse provider
provider = create_embedding_provider("openai")
for batch in text_batches:
    embeddings = await provider.generate_embeddings_batch(batch)

# ❌ Avoid - Creating new providers repeatedly
for batch in text_batches:
    provider = create_embedding_provider("openai")
    embeddings = await provider.generate_embeddings_batch(batch)
```

## 🔍 Common Use Cases

### Semantic Search

```python
import numpy as np

provider = create_embedding_provider(
    task_type=EmbeddingTaskType.SEMANTIC_SIMILARITY
)

# Embed documents
docs = ["doc1", "doc2", "doc3"]
doc_embeddings = await provider.generate_embeddings_batch(docs)

# Embed query
query = "search query"
query_embedding = await provider.generate_embedding(query)

# Find most similar
similarities = [
    np.dot(query_embedding, doc_emb)
    for doc_emb in doc_embeddings
]
best_match_idx = np.argmax(similarities)
```

### Document Clustering

```python
provider = create_embedding_provider(
    task_type=EmbeddingTaskType.CLUSTERING
)

# Generate embeddings
documents = ["doc1", "doc2", "doc3"]
embeddings = await provider.generate_embeddings_batch(documents)

# Use with clustering algorithm (e.g., K-means)
from sklearn.cluster import KMeans
clusters = KMeans(n_clusters=2).fit_predict(embeddings)
```

### Classification

```python
provider = create_embedding_provider(
    task_type=EmbeddingTaskType.CLASSIFICATION
)

# Generate embeddings for training data
texts = ["positive example", "negative example"]
embeddings = await provider.generate_embeddings_batch(texts)

# Train classifier on embeddings
```

## 📚 Additional Resources

- [OpenAI Embeddings Documentation](https://platform.openai.com/docs/guides/embeddings)
- [Gemini Embeddings Documentation](https://ai.google.dev/docs/embeddings_guide)
- [Usage Examples](./embedding_usage_examples.py) - Comprehensive code examples
- [Exception Reference](./embedding_exceptions.py) - All exception types

## 🆘 Support

For issues and questions:
- Check [embedding_usage_examples.py](./embedding_usage_examples.py) for common patterns
- Review exception types in [embedding_exceptions.py](./embedding_exceptions.py)
- Enable debug logging: `logger.setLevel(logging.DEBUG)`

---

**Version**: 2.0.0  
**Last Updated**: 2025