"""Check the local services the healthcare agent depends on."""

from src.adapters.redis import RedisProvider
from src.adapters.redis.exceptions import CacheProviderError
from src.adapters.vector_store.exceptions import VectorDBProviderError
from src.adapters.vector_store.milvus_vector_store import MilvusProvider
from src.drugs_agent import verify_drug_collection_ready_for_search
from src.logs import get_component_logger

logger = get_component_logger("LocalServices")


async def check_redis(provider: RedisProvider) -> bool:
    """Connect to Redis and verify its PING response."""
    try:
        await provider.connect()
    except CacheProviderError as exc:
        logger.warning("Redis is not ready", error=str(exc))
        return False
    return True


async def check_milvus(provider: MilvusProvider) -> bool:
    """Connect to Milvus and confirm the drug collection holds data."""
    try:
        count = await verify_drug_collection_ready_for_search(provider)
    except VectorDBProviderError as exc:
        logger.warning("Milvus is not ready", error=str(exc))
        await provider.disconnect()
        return False
    logger.info("Drug collection is ready", rows=count)
    return True
