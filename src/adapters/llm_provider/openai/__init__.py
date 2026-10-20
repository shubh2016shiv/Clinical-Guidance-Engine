"""OpenAI implementation of the LLM provider contract.

``provider.py`` is the entry point; ``chat_completions.py`` and
``responses_api.py`` own the two OpenAI API paradigms, and ``embeddings.py``
creates text embeddings.
"""

from .embeddings import OpenAIEmbeddings
from .provider import OpenAIProvider

__all__ = ["OpenAIEmbeddings", "OpenAIProvider"]
