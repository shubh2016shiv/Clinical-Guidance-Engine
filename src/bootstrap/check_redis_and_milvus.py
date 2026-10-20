"""Check local services used by the healthcare agent."""

from src.logs import get_component_logger
from src.response_api_agent.asclepius_agent import AsclepiusHealthcareAgent

logger = get_component_logger("LocalServices")


async def check_redis(agent: AsclepiusHealthcareAgent) -> bool:
    """Connect to Redis and verify its PING response."""
    ready = await agent.response_manager.connect_cache()
    if not ready:
        agent.cache_provider = None
    return ready


async def check_milvus(agent: AsclepiusHealthcareAgent) -> bool:
    """Connect to Milvus and load the configured drug collection."""
    manager = agent.response_manager
    drug_data = manager.drug_data_manager
    if drug_data is None:
        return False
    try:
        await drug_data.connect()
        await drug_data.vector_db_provider.check_collection()
        return True
    except Exception as exc:
        logger.warning("Milvus is not ready", error=str(exc))
        await drug_data.disconnect()
        manager.drug_data_manager = None
        return False
