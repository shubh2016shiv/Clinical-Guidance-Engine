"""Redis connection, cache lifetime, and retry settings."""

from pydantic import BaseModel, Field


class RedisSettings(BaseModel):
    """Fields for the conversation cache and its connection behavior."""

    redis_host: str = Field(
        default="localhost",
        description="Redis server host. Use 'redis' inside Docker Compose network.",
    )

    redis_port: int = Field(
        default=6381,
        ge=1,
        le=65535,
        description="Redis server port. 6381 is the host port docker-compose.yml publishes "
        "for this project's Redis.",
    )

    redis_db: int = Field(
        default=0,
        ge=0,
        le=15,
        description="Redis database index (0–15).",
    )

    redis_password: str = Field(
        default="redis123",
        description="Redis authentication password.",
    )

    redis_max_connections: int = Field(
        default=50,
        ge=1,
        description="Maximum connections in the Redis connection pool.",
    )

    redis_socket_timeout: int = Field(
        default=5,
        ge=1,
        description="Redis socket timeout in seconds.",
    )

    redis_key_prefix: str = Field(
        default="Asclepius_Conversations",
        description="Prefix for all Redis keys (namespace isolation).",
    )

    redis_default_session_ttl: int = Field(
        default=7200,
        ge=0,
        description="Default TTL for session data in seconds (7200 = 2 hours).",
    )

    redis_default_history_ttl: int = Field(
        default=1800,
        ge=0,
        description="Default TTL for chat-history cache in seconds (1800 = 30 minutes).",
    )

    # Resilience settings
    redis_circuit_breaker_fail_max: int = Field(
        default=5,
        ge=1,
        description="Number of consecutive failures before the Redis circuit breaker opens.",
    )

    redis_circuit_breaker_timeout: int = Field(
        default=30,
        ge=1,
        description="Seconds the Redis circuit breaker stays open before attempting half-open recovery.",
    )

    redis_retry_max_attempts: int = Field(
        default=3,
        ge=0,
        description="Maximum retry attempts for transient Redis operation failures. 0 = no retries.",
    )

    redis_retry_base_delay: float = Field(
        default=0.1,
        ge=0.0,
        description="Base delay in seconds for exponential backoff between Redis retries.",
    )

    redis_retry_max_delay: float = Field(
        default=2.0,
        ge=0.0,
        description="Maximum delay in seconds for exponential backoff between Redis retries.",
    )
