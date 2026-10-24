"""OpenAI API, response model, and embedding settings.

Big-picture flow:
1. Describe the credentials and model name needed to call OpenAI.
2. Describe how "creative" vs. "predictable" the model's answers should be.
3. Describe the embedding model used to turn drug text into vectors for
   search - this MUST match whatever the data-ingestion pipeline used.
"""

from pydantic import BaseModel, Field

# See paths.py for what Pydantic's BaseModel and Field do.


# OpenAISettings: everything needed to call OpenAI's APIs, grouped in one
# place so the LLM provider adapter (src/adapters/llm_provider) can read it
# without needing to know about Redis, Milvus, or any other service.
class OpenAISettings(BaseModel):
    """Fields used by OpenAI clients and the drug embedding pipeline."""

    openai_api_key: str = Field(
        default="",
        description="OpenAI API Key. Required for OpenAI API access.",
    )
    # openai_api_key: the secret credential OpenAI uses to identify and
    # bill requests. An empty default means "not set yet" - see
    # check_settings.py in src/bootstrap, which rejects a blank key.

    openai_model_name: str = Field(
        default="gpt-4o",
        description="OpenAI model name (e.g. gpt-4o, gpt-4o-mini).",
    )
    # openai_model_name: which OpenAI model to call for chat answers.

    openai_temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (0.0 = deterministic, 2.0 = max creativity).",
    )
    # openai_temperature: how random the model's wording is. "ge"/"le"
    # (greater-than-or-equal / less-than-or-equal) tell Pydantic to reject
    # any value outside 0.0-2.0 automatically, so a typo like 20 never
    # reaches OpenAI.

    openai_top_p: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling parameter. Some Responses API models may not support this.",
    )
    # openai_top_p: an alternative way (besides temperature) to control how
    # varied the model's wording is.

    openai_max_output_tokens: int = Field(
        default=2000,
        ge=1,
        description="Maximum response length in tokens.",
    )
    # openai_max_output_tokens: the longest answer the model is allowed to
    # generate, measured in tokens (roughly, pieces of words).

    openai_embedding_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model. Must match the model used by the ingestion pipeline.",
    )
    # openai_embedding_model: which model turns text into a vector (a list
    # of numbers) for similarity search. NOTE: if this does not match the
    # model used when the Milvus drug collection was originally filled,
    # drug search will silently return wrong or no results.
