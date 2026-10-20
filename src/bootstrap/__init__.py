"""Coordinate startup checks and return a ready healthcare agent."""

from dataclasses import dataclass, field

from src.config import get_settings
from src.logs import get_component_logger
from src.response_api_agent.asclepius_agent import AsclepiusHealthcareAgent

from .check_redis_and_milvus import check_milvus, check_redis
from .check_settings import check_settings
from .prepare_guidelines_store import prepare_guidelines_store


@dataclass
class BootstrapResult:
    agent: AsclepiusHealthcareAgent
    vector_store_id: str | None = None
    cache_ready: bool = False
    drug_search_ready: bool = False
    guidelines_ready: bool = False
    warnings: list[str] = field(default_factory=list)


async def bootstrap_agent(
    *,
    chat_history_limit: int = 10,
    use_drug_database: bool = True,
    use_clinical_guidelines: bool = True,
) -> BootstrapResult:
    """Construct the agent and check its optional dependencies.

    Docker services must already be running. Failed optional services are
    disabled and recorded as warnings, so basic chat can still work.
    """
    check_settings(
        get_settings(),
        use_drug_database=use_drug_database,
        use_clinical_guidelines=use_clinical_guidelines,
    )
    logger = get_component_logger("Bootstrap")
    agent = AsclepiusHealthcareAgent(chat_history_limit=chat_history_limit)
    result = BootstrapResult(agent=agent)

    result.cache_ready = await check_redis(agent)
    if not result.cache_ready:
        result.warnings.append("Redis is unavailable; session caching is disabled")

    if use_drug_database:
        result.drug_search_ready = await check_milvus(agent)
        if not result.drug_search_ready:
            result.warnings.append("Drug search is unavailable; check Milvus and run ingestion")
    else:
        agent.response_manager.drug_data_manager = None

    if use_clinical_guidelines:
        result.vector_store_id = await prepare_guidelines_store(agent)
        result.guidelines_ready = result.vector_store_id is not None
        if not result.guidelines_ready:
            result.warnings.append("Clinical guidelines are not ready; file search is disabled")

    for warning in result.warnings:
        logger.warning(warning)
    return result


__all__ = ["BootstrapResult", "bootstrap_agent"]
