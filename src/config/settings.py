"""Load the application's flat environment settings from service-specific fields.

Big-picture flow:
1. Combine every service-specific settings group (OpenAI, Milvus, Redis,
   Guidelines, Application, Paths) into one big ``Settings`` class.
2. Load real values from environment variables / the .env file, and
   validate them using Pydantic's rules (the "ge"/"le" checks, etc. seen in
   the other files in this folder).
3. Cache the loaded settings so the whole app reuses ONE validated
   instance instead of re-reading the .env file every time settings are
   needed.
4. Log what was loaded (with secrets masked) so a developer can see the
   active configuration without it leaking credentials into logs.
"""

import sys
from functools import lru_cache

# lru_cache is a decorator that remembers a function's past results, keyed
# by its arguments, and returns the cached result instead of running the
# function again. Below, it is used with maxsize=1 to turn get_settings()
# into "compute once, reuse forever" for the lifetime of the program.
from typing import Any, ClassVar

# "Any" is a type hint meaning "could be any type". ClassVar marks an
# attribute as belonging to the CLASS itself, not to each individual
# instance, and tells Pydantic not to treat it as a settings field.
from pydantic import ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

# BaseSettings is Pydantic's special base class for settings that get their
# values from environment variables and .env files (instead of being passed
# in directly by code). SettingsConfigDict configures HOW that loading
# behaves (which file, case sensitivity, etc.).
from .application_settings import ApplicationSettings
from .guidelines_settings import GuidelinesSettings
from .milvus_settings import MilvusSettings
from .openai_settings import OpenAISettings
from .paths import ENV_FILE, PathSettings
from .redis_settings import RedisSettings


# Settings: the one class the rest of the app actually uses. It combines
# every other settings group in this folder into a single object, so code
# elsewhere can just write "settings.redis_host" or "settings.openai_api_key"
# without caring which file originally defined that field.
class Settings(
    BaseSettings,
    OpenAISettings,
    MilvusSettings,
    RedisSettings,
    GuidelinesSettings,
    ApplicationSettings,
    PathSettings,
):
    # Listing several classes in parentheses is "multiple inheritance":
    # Settings becomes a combination of ALL of them at once, inheriting
    # every field each one defines. Think of each settings class as one
    # ingredient, and Settings as the final dish made from all of them.
    # BaseSettings specifically is what makes the combined class capable of
    # reading values from environment variables and the .env file.
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
    # model_config: settings ABOUT how Settings itself behaves (not an app
    # setting). In short:
    # - env_file: read values from this .env file if present.
    # - case_sensitive=False: "OPENAI_API_KEY" and "openai_api_key" match.
    # - extra="forbid": an unknown environment variable name that looks
    #   like it was meant for this app raises an error instead of being
    #   silently ignored - this catches typos in variable names early.
    # - frozen=True: once created, a Settings instance cannot be changed -
    #   this prevents some part of the app from accidentally mutating
    #   shared configuration at runtime.
    # - validate_default=True: even the DEFAULT values declared with Field()
    #   get validated (e.g. checked against ge/le), not just values that
    #   come from the environment.
    # - env_parse_none_str="null": lets an environment variable literally
    #   containing the text "null" be treated as Python's None.

    _SECRET_FIELDS: ClassVar[set[str]] = {
        "openai_api_key",
        "milvus_password",
        "redis_password",
        "minio_root_password",
    }
    # _SECRET_FIELDS: the list of setting names that hold passwords/keys.
    # Used below so they can be partially hidden in logs instead of printed
    # in full.

    def log_loaded_settings(self, *, logger: Any = None) -> None:
        """Log loaded fields while masking credentials."""
        # The "*" makes "logger" keyword-only, so callers must write
        # "log_loaded_settings(logger=...)" rather than relying on position.
        # Step 1: decide HOW to log - use the given logger's .debug method
        # if one was passed in, otherwise fall back to a tiny throwaway
        # function that just calls print(). "lambda message, **_: print(message)"
        # is an unnamed, one-line function: it takes a message plus any
        # extra keyword arguments (collected into "_" and ignored) and
        # prints just the message - this lets the same call below work
        # whether or not a real logger is available.
        log_fn = logger.debug if logger is not None else lambda message, **_: print(message)
        for name in sorted(type(self).model_fields):
            # Step 2: loop over every declared setting NAME (sorted so the
            # log output is in a stable, predictable order every run).
            # "type(self).model_fields" is Pydantic's dictionary of every
            # field this class declares, across all the combined settings
            # groups.
            raw_value = getattr(self, name)
            # "getattr(self, name)" reads the attribute whose NAME is the
            # string in "name" (e.g. "redis_host") - a dynamic version of
            # writing "self.redis_host" by hand, needed here because we are
            # looping over field names instead of listing them one by one.
            value = _mask_secret(str(raw_value)) if name in self._SECRET_FIELDS else raw_value
            # Step 3: if this field is one of the secret ones, mask it
            # before logging; otherwise log it as-is.
            log_fn(
                "Settings loaded",
                component="Config",
                subcomponent="Settings",
                field=name,
                value=value,
            )
            # Step 4: emit one log line per setting, naming the field and
            # its (possibly masked) value.


# _mask_secret: hides most of a secret value so it can still be logged for
# debugging (e.g. "did the key actually load?") without exposing the whole
# thing. The leading underscore signals "internal to this file only".
def _mask_secret(value: str) -> str:
    if not value:
        # Step 1: nothing to mask if the value is empty - say so plainly.
        return "<not set>"
    if len(value) <= 8:
        # Step 2: a very short secret would reveal too much if we showed
        # even a few real characters, so replace the whole thing with
        # asterisks of the same length.
        return "*" * len(value)
    # Step 3: for anything longer, show just the first and last 4
    # characters with "****" in between - enough to recognize "is this the
    # key I expect?" without exposing the actual secret.
    return value[:4] + "****" + value[-4:]


# _emit_warning: prints a warning to stderr (the "error" output stream)
# instead of stdout, so warnings are visible even if normal program output
# is being redirected elsewhere (for example, into a file).
def _emit_warning(message: str) -> None:
    print(f"[WARNING ] {message}", file=sys.stderr)


# _check_env_file_exists: warns early if the expected .env file is simply
# missing, which is the most common reason settings fail to load - this
# turns a confusing validation error later into a clear, actionable hint
# right away.
def _check_env_file_exists() -> None:
    if not ENV_FILE.is_file():
        _emit_warning(
            f"Environment file not found at {ENV_FILE}. "
            "Copy env_example.txt to .env and fill in your API keys, "
            "or export the required variables in your shell."
        )


# get_settings: the ONE function the rest of the app should call to get
# configuration - never create "Settings()" directly elsewhere, so there is
# always exactly one validated, cached settings object shared everywhere.
@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the shared settings instance, loaded and validated once."""
    # "@lru_cache(maxsize=1)" means: the FIRST time this function runs, it
    # actually loads and validates settings; every call after that (with no
    # arguments, so there is only ever one possible cache entry) instantly
    # returns the same already-built Settings object instead of redoing the
    # work.
    # Step 1: give an early, friendly hint if the .env file is missing,
    # before attempting the stricter validation below.
    _check_env_file_exists()
    try:
        # Step 2: this is where environment variables / the .env file are
        # actually read and checked against every Field's rules (ge, le,
        # required, etc.) across all the combined settings groups.
        settings = Settings()
    except ValidationError as error:
        # Step 3: if something is missing or invalid, Pydantic raises a
        # ValidationError listing every problem at once. Print each one in
        # a readable way before re-raising, so a developer sees exactly
        # what to fix instead of a raw stack trace.
        _emit_warning(f"Settings validation failed ({error.error_count()} error(s)):")
        for detail in error.errors():
            _emit_warning(f"  - {detail['loc']}: {detail['msg']}")
        raise
        # "raise" with no argument here re-raises the SAME exception that
        # was just caught, after we have printed our extra context - the
        # caller still sees the failure and the program still stops.

    try:
        # Step 4: try to log the loaded settings using the app's real
        # logger. This is wrapped in try/except because the logging system
        # itself depends on settings (a small chicken-and-egg problem), so
        # importing it here could fail in some edge cases.
        from src.logs import get_component_logger

        settings.log_loaded_settings(logger=get_component_logger("Config"))
    except Exception:
        # Step 5: if real logging is not available for any reason, fall
        # back to plain print() - but ONLY if debug_mode is on, so normal
        # runs are not spammed with every setting on every start-up.
        if settings.debug_mode:
            settings.log_loaded_settings(logger=None)
    return settings
