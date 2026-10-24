"""Application metadata and runtime feature settings.

Big-picture flow:
1. Describe facts ABOUT the app itself (its name, version) rather than
   about any external service it talks to.
2. Describe a few on/off switches (feature flags) that change how the app
   behaves at runtime, like whether it streams answers or logs verbosely.
"""

from pydantic import BaseModel, Field

# See paths.py for what Pydantic's BaseModel and Field do. In short:
# BaseModel groups related settings together, and Field gives each one a
# default value plus a description used in error messages and docs.


# ApplicationSettings: groups settings that describe the app itself, kept
# separate from service-specific settings (OpenAI, Redis, etc.) so this
# file stays easy to scan when someone just wants to change app metadata
# or toggle a feature.
class ApplicationSettings(BaseModel):
    """Fields that describe the app rather than an external service."""

    app_name: str = Field(
        default="Drug Recommendation Chatbot",
        description="Human-readable application name.",
    )
    # app_name: shown in logs/UI; purely descriptive, does not affect behavior.

    app_version: str = Field(
        default="1.0.0",
        description="Application version (semver).",
    )
    # app_version: a version label for this build of the app.

    debug_mode: bool = Field(
        default=False,
        description="Enable debug mode (verbose logging, dev helpers).",
    )
    # debug_mode: when True, other parts of the app may log more detail or
    # enable developer-only behavior (see settings.py's get_settings()).

    enable_streaming: bool = Field(
        default=True,
        description="Global toggle for streaming (SSE) responses.",
    )
    # enable_streaming: when True, answers can be sent to the user piece by
    # piece as they are generated, instead of all at once.

    enable_cleanup: bool = Field(
        default=False,
        description="Enable resource cleanup on agent shutdown.",
    )
    # enable_cleanup: when True, agents release their resources (like open
    # connections) when shutting down.
