"""Expose the OpenAI tools used by the healthcare assistant.

This package keeps OpenAI-specific code behind a small import path.
1. ``provider.py`` chooses Chat Completions or the Responses API.
2. Each API client sends its own kind of request.
3. ``embeddings.py`` turns search text into numbers for vector search.
"""

# These imports let callers reach the public classes without knowing file names.
from .embeddings import OpenAITextEmbeddingClient
from .provider import OpenAIProvider

# __all__ identifies the public classes exported by this package.
__all__ = ["OpenAITextEmbeddingClient", "OpenAIProvider"]
