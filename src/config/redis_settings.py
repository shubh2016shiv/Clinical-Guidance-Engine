"""Redis connection, cache lifetime, and retry settings.

Big-picture flow:
1. Describe where the Redis server is and how to authenticate to it.
2. Describe how long cached data should live before expiring (TTL values).
3. Describe resilience settings: how the app reacts to Redis being slow or
   down, using a "circuit breaker" and retry backoff.
"""

from pydantic import BaseModel, Field

# See paths.py for what Pydantic's BaseModel and Field do.


# RedisSettings: groups everything the Redis cache adapter needs - where
# to connect, how long to keep things cached, and how to behave if Redis
# is slow or unreachable.
class RedisSettings(BaseModel):
    """Fields for the conversation cache and its connection behavior."""

    redis_host: str = Field(
        default="localhost",
        description="Redis server host. Use 'redis' inside Docker Compose network.",
    )
    # redis_host: where to find the Redis server. "localhost" when running
    # outside Docker; the service name "redis" when running inside the Docker
    # Compose network, since containers address each other by service name
    # instead of "localhost".

    redis_port: int = Field(
        default=6381,
        ge=1,
        le=65535,
        description="Redis server port. 6381 is the host port docker-compose.yml publishes "
        "for this project's Redis.",
    )
    # redis_port: which network port Redis is listening on.

    redis_db: int = Field(
        default=0,
        ge=0,
        le=15,
        description="Redis database index (0–15).",
    )
    # redis_db: Redis keeps 16 separate numbered databases in one server;
    # this picks which one this app uses.

    redis_password: str = Field(
        default="redis123",
        description="Redis authentication password.",
    )
    # redis_password: NOTE: this default is a local-development placeholder
    # and should be overridden for any real deployment.

    redis_max_connections: int = Field(
        default=50,
        ge=1,
        description="Maximum connections in the Redis connection pool.",
    )
    # redis_max_connections: a "pool" is a set of ready-to-use connections
    # shared between requests, so the app does not open a brand-new
    # connection every single time. This caps how many it may open at once.

    redis_socket_timeout: int = Field(
        default=5,
        ge=1,
        description="Redis socket timeout in seconds.",
    )
    # redis_socket_timeout: how long to wait for Redis to respond before
    # giving up on a single operation.

    redis_socket_connect_timeout: int = Field(
        default=5,
        ge=1,
        description="Timeout in seconds for opening a Redis connection.",
    )
    # redis_socket_connect_timeout: fail promptly when Redis cannot be reached.

    redis_ssl: bool = Field(
        default=False,
        description="Use TLS when connecting to Redis.",
    )
    # redis_ssl: enable encrypted connections for Redis deployments that require TLS.

    redis_key_prefix: str = Field(
        default="healthcare_conversations",
        description="Prefix for all Redis keys (namespace isolation).",
    )
    # redis_key_prefix: text added to the front of every key this app
    # stores, so its data does not collide with another app's data if they
    # happen to share the same Redis server.

    redis_default_session_ttl: int = Field(
        default=7200,
        ge=1,
        description="Default TTL for session data in seconds (7200 = 2 hours).",
    )
    # redis_default_session_ttl: how long a user's session stays cached
    # before Redis automatically deletes it.

    redis_default_history_ttl: int = Field(
        default=1800,
        ge=1,
        description="Default TTL for chat-history cache in seconds (1800 = 30 minutes).",
    )
    # redis_default_history_ttl: how long cached chat history stays around
    # before Redis automatically deletes it.

    # Resilience settings
    redis_circuit_breaker_fail_max: int = Field(
        default=5,
        ge=1,
        description="Number of consecutive failures before the Redis circuit breaker opens.",
    )
    # redis_circuit_breaker_fail_max: a "circuit breaker" stops trying to
    # use a failing service for a while, the same way an electrical circuit
    # breaker trips to stop a surge - this is how many failures in a row it
    # takes before the breaker "opens" (stops trying).

    redis_circuit_breaker_timeout: int = Field(
        default=30,
        ge=1,
        description="Seconds the Redis circuit breaker stays open before attempting half-open recovery.",
    )
    # redis_circuit_breaker_timeout: how long the breaker stays open before
    # cautiously trying Redis again ("half-open") to see if it has recovered.

    redis_retry_max_attempts: int = Field(
        default=3,
        ge=0,
        description="Maximum retry attempts for transient Redis operation failures. 0 = no retries.",
    )
    # redis_retry_max_attempts: how many times to retry a failed Redis
    # operation before giving up.

    redis_retry_base_delay: float = Field(
        default=0.1,
        ge=0.0,
        description="Base delay in seconds for exponential backoff between Redis retries.",
    )
    # redis_retry_base_delay: "exponential backoff" means waiting longer and
    # longer between retries (e.g. 0.1s, 0.2s, 0.4s...) instead of retrying
    # instantly, to give Redis time to recover. This is the starting delay.

    redis_retry_max_delay: float = Field(
        default=2.0,
        ge=0.0,
        description="Maximum delay in seconds for exponential backoff between Redis retries.",
    )
    # redis_retry_max_delay: the longest the backoff delay is allowed to
    # grow to, even if more retries are needed.
