"""Check startup dependencies and return the services that are ready to use."""

from dataclasses import dataclass, field

from src.adapters.redis import CacheConfig, RedisProvider
from src.adapters.vector_store.milvus_vector_store import MilvusProvider
from src.clinical_guideline_agent import ClinicalGuidelineAgent
from src.config import get_settings
from src.logs import get_component_logger

from .check_redis_and_milvus import check_milvus, check_redis
from .check_settings import check_settings
from .prepare_guidelines_store import prepare_guidelines_store


@dataclass
class BootstrapResult:
    cache: RedisProvider | None = None
    drug_provider: MilvusProvider | None = None
    guideline_agent: ClinicalGuidelineAgent | None = None
    warnings: list[str] = field(default_factory=list)

    async def close(self) -> None:
        if self.cache is not None:
            await self.cache.disconnect()
        if self.drug_provider is not None:
            await self.drug_provider.disconnect()
        if self.guideline_agent is not None:
            await self.guideline_agent.close()


async def bootstrap_services(
    *,
    use_drug_database: bool = True,
    use_clinical_guidelines: bool = True,
) -> BootstrapResult:
    """Connect to each service and keep only the ones that are ready.

    Docker services must already be running. A service that fails its check is left
    out of the result and recorded as a warning, so basic chat can still work.
    """
    settings = get_settings()
    check_settings(
        settings,
        use_drug_database=use_drug_database,
        use_clinical_guidelines=use_clinical_guidelines,
    )
    logger = get_component_logger("Bootstrap")
    result = BootstrapResult()

    cache = RedisProvider(CacheConfig.from_settings(settings))
    if await check_redis(cache):
        result.cache = cache
    else:
        result.warnings.append("Redis is unavailable; session caching is disabled")

    if use_drug_database:
        drug_provider = MilvusProvider()
        if await check_milvus(drug_provider):
            result.drug_provider = drug_provider
        else:
            result.warnings.append("Drug search is unavailable; check Milvus and run ingestion")

    if use_clinical_guidelines:
        guideline_agent = ClinicalGuidelineAgent()
        if await prepare_guidelines_store(guideline_agent):
            result.guideline_agent = guideline_agent
        else:
            await guideline_agent.close()
            result.warnings.append("Clinical guidelines are not ready; file search is disabled")

    for warning in result.warnings:
        logger.warning(warning)
    return result


__all__ = ["BootstrapResult", "bootstrap_services"]
