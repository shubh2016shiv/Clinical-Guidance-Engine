"""Project paths and Docker volume location."""

from pathlib import Path

from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"


class PathSettings(BaseModel):
    docker_volume_directory: str = Field(
        default=".",
        description="Base directory for Docker Compose persistent volumes. "
        "Defaults to the project root ('.').",
    )
