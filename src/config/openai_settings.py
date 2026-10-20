"""OpenAI API, response model, and embedding settings."""

from pydantic import BaseModel, Field


class OpenAISettings(BaseModel):
    """Fields used by OpenAI clients and the drug embedding pipeline."""

    openai_api_key: str = Field(
        default="",
        description="OpenAI API Key. Required for OpenAI API access.",
    )

    openai_model_name: str = Field(
        default="gpt-4o",
        description="OpenAI model name (e.g. gpt-4o, gpt-4o-mini).",
    )

    openai_temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (0.0 = deterministic, 2.0 = max creativity).",
    )

    openai_top_p: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling parameter. Some Responses API models may not support this.",
    )

    openai_max_output_tokens: int = Field(
        default=2000,
        ge=1,
        description="Maximum response length in tokens.",
    )

    openai_embedding_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model. Must match the model used by the ingestion pipeline.",
    )
