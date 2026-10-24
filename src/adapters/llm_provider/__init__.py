"""Give the rest of the healthcare assistant one place to import model tools.

This package hides the details of calling a language model from its callers.
1. Define shared settings, results, and errors in the nearby modules.
2. Expose the OpenAI implementation through this package.
3. Keep Chat Completions results and Responses API SDK results available.

Beginner note: this file is an "__init__.py". Python treats a folder
containing one of these as a package, and whatever this file imports
becomes available directly from the package's name, e.g.
``from src.adapters.llm_provider import OpenAIProvider`` instead of the
longer ``from src.adapters.llm_provider.openai.provider import OpenAIProvider``.
"""

# These imports make the shared types available from this package's main path.
from .base import APIType, LLMConfig, LLMProvider, LLMResponse, StreamMode
from .openai import OpenAIProvider

# __all__ lists the names meant for other modules to import from here.
# It also controls what "from src.adapters.llm_provider import *" brings in.
__all__ = ["APIType", "LLMConfig", "LLMProvider", "LLMResponse", "StreamMode", "OpenAIProvider"]
