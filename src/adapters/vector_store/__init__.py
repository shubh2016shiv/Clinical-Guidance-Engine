"""Vector database adapter: Milvus for drug search, OpenAI for guideline files.

Beginner note: this file is this package's "__init__.py", so whatever it
imports becomes available directly as ``from src.adapters.vector_store
import MilvusProvider`` instead of the longer
``from src.adapters.vector_store.milvus_vector_store import MilvusProvider``.
"""

from .base import SearchResult, VectorDBProvider
from .milvus_vector_store import MilvusProvider
from .openai_vector_store import OpenAIVectorStoreClient

# __all__ lists the names meant for other modules to import from here.
__all__ = ["MilvusProvider", "OpenAIVectorStoreClient", "SearchResult", "VectorDBProvider"]
