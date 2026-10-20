"""Application metadata and runtime feature settings."""

from pydantic import BaseModel, Field


class ApplicationSettings(BaseModel):
    """Fields that describe the app rather than an external service."""

    app_name: str = Field(
        default="Drug Recommendation Chatbot",
        description="Human-readable application name.",
    )

    app_version: str = Field(
        default="1.0.0",
        description="Application version (semver).",
    )

    debug_mode: bool = Field(
        default=False,
        description="Enable debug mode (verbose logging, dev helpers).",
    )

    enable_streaming: bool = Field(
        default=True,
        description="Global toggle for streaming (SSE) responses.",
    )

    enable_cleanup: bool = Field(
        default=False,
        description="Enable resource cleanup on agent shutdown.",
    )
