"""Check the local services the healthcare agent depends on.

Big-picture flow:
1. Try to connect to each service (Redis, Milvus).
2. If the connection or a basic health check fails, log a warning and
   report "not ready" (return False) instead of crashing the whole app.
3. The caller (see __init__.py in this package) decides what to do when a
   service is not ready - usually, turn that one feature off and keep going.
"""

from src.adapters.redis import RedisProvider
from src.adapters.redis.exceptions import CacheProviderError
from src.adapters.vector_store.exceptions import VectorDBProviderError
from src.adapters.vector_store.milvus_vector_store import MilvusProvider
from src.drugs_agent import verify_drug_collection_ready_for_search
from src.logs import get_component_logger

logger = get_component_logger("LocalServices")
# A logger lets this file record warnings (like "Redis is not ready") that
# show up in the app's logs, so a developer can diagnose startup problems
# without the app needing to crash to reveal them.


# check_redis: confirms the cache is reachable before the rest of the app
# relies on it. Needed because Redis is optional at startup - if it is
# down, the app should still be able to answer questions, just without
# cached chat history.
async def check_redis(provider: RedisProvider) -> bool:
    """Connect to Redis and verify its PING response."""
    # "async def" means this function does network I/O and must be awaited;
    # it can pause here while waiting for Redis to respond.
    try:
        # Step 1: try to open the connection (this also sends a PING under
        # the hood to make sure Redis is actually responding, not just that
        # the network path exists).
        await provider.connect()
    except CacheProviderError as exc:
        # Step 2: if Redis refuses the connection or the PING fails, this is
        # a KNOWN, expected kind of failure (not a bug) - log it clearly and
        # tell the caller "this service is not ready" instead of raising.
        logger.warning("Redis is not ready", error=str(exc))
        return False
    # Step 3: no exception means Redis is reachable and healthy.
    return True


# check_milvus: confirms the vector database is reachable AND already has
# drug data loaded into it. Needed because an empty or unreachable Milvus
# would otherwise let drug search run and quietly return nothing useful.
async def check_milvus(provider: MilvusProvider) -> bool:
    """Connect to Milvus and confirm the drug collection holds data."""
    try:
        # Step 1: this helper both connects AND checks that the drug
        # collection actually has rows in it, returning that row count.
        count = await verify_drug_collection_ready_for_search(provider)
    except VectorDBProviderError as exc:
        # Step 2: if Milvus is down or the collection is missing/empty,
        # clean up the connection we may have opened, log a clear warning,
        # and report "not ready" rather than crashing startup.
        logger.warning("Milvus is not ready", error=str(exc))
        await provider.disconnect()
        return False
    # Step 3: log the row count on success - useful for confirming
    # ingestion actually populated the collection as expected.
    logger.info("Drug collection is ready", rows=count)
    return True
