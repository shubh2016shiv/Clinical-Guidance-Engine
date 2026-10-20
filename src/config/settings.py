"""Load the application's flat environment settings from service-specific fields."""

import sys
from functools import lru_cache
from typing import Any, ClassVar

from pydantic import ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

from .application_settings import ApplicationSettings
from .guidelines_settings import GuidelinesSettings
from .milvus_settings import MilvusSettings
from .mongodb_settings import MongoDBSettings
from .openai_settings import OpenAISettings
from .paths import ENV_FILE, PathSettings
from .redis_settings import RedisSettings


class Settings(
    BaseSettings,
    OpenAISettings,
    MilvusSettings,
    RedisSettings,
    MongoDBSettings,
    GuidelinesSettings,
    ApplicationSettings,
    PathSettings,
):
    """One validated, immutable view of the existing flat environment variables."""

    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="forbid",
        frozen=True,
        validate_default=True,
        env_parse_none_str="null",
    )

    _SECRET_FIELDS: ClassVar[set[str]] = {
        "openai_api_key",
        "milvus_password",
        "redis_password",
        "mongo_root_password",
        "minio_root_password",
    }

    def log_loaded_settings(self, *, logger: Any = None) -> None:
        """Log loaded fields while masking credentials."""
        log_fn = logger.debug if logger is not None else lambda message, **_: print(message)
        for name in sorted(type(self).model_fields):
            raw_value = getattr(self, name)
            value = _mask_secret(str(raw_value)) if name in self._SECRET_FIELDS else raw_value
            log_fn(
                "Settings loaded",
                component="Config",
                subcomponent="Settings",
                field=name,
                value=value,
            )


def _mask_secret(value: str) -> str:
    if not value:
        return "<not set>"
    if len(value) <= 8:
        return "*" * len(value)
    return value[:4] + "****" + value[-4:]


def _emit_warning(message: str) -> None:
    print(f"[WARNING ] {message}", file=sys.stderr)


def _check_env_file_exists() -> None:
    if not ENV_FILE.is_file():
        _emit_warning(
            f"Environment file not found at {ENV_FILE}. "
            "Copy env_example.txt to .env and fill in your API keys, "
            "or export the required variables in your shell."
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the shared settings instance, loaded and validated once."""
    _check_env_file_exists()
    try:
        settings = Settings()
    except ValidationError as error:
        _emit_warning(f"Settings validation failed ({error.error_count()} error(s)):")
        for detail in error.errors():
            _emit_warning(f"  - {detail['loc']}: {detail['msg']}")
        raise

    try:
        from src.logs import get_component_logger

        settings.log_loaded_settings(logger=get_component_logger("Config"))
    except Exception:
        if settings.debug_mode:
            settings.log_loaded_settings(logger=None)
    return settings
