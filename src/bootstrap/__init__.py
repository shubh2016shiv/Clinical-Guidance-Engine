"""Check startup dependencies and return the services that are ready to use.

Big-picture flow:
1. Validate settings up front (see check_settings.py) so missing config
   fails loudly and immediately, before anything tries to connect.
2. Try to connect to each optional service: Redis cache, Milvus drug search,
   and the clinical guidelines store.
3. For each one that fails, record a warning and leave it out of the
   result, instead of crashing the whole app - the chatbot should still be
   able to answer basic questions even if, say, Redis is down.
4. Hand back a single ``BootstrapResult`` the rest of the app can use to
   see exactly which features are available this run.

Beginner note: this file is this package's "__init__.py", so everything it
defines is importable directly as ``from src.bootstrap import ...`` instead
of needing the longer ``from src.bootstrap.<filename> import ...`` path.
"""

from dataclasses import dataclass, field

# @dataclass is a shortcut that writes the boring parts of a class (like
# __init__) for us, just from the fields listed below. "field" lets us give
# a field a safer default than writing "= []" directly (see the note below).
from src.adapters.redis import RedisCacheConfig, RedisProvider
from src.adapters.vector_store.milvus_vector_store import MilvusProvider
from src.clinical_guideline_agent import ClinicalGuidelineAgent
from src.config import get_settings
from src.logs import get_component_logger

from .check_redis_and_milvus import check_milvus, check_redis
from .check_settings import check_settings
from .prepare_guidelines_store import prepare_guidelines_store


# BootstrapResult: a single container holding whichever services started up
# successfully, plus any warnings about ones that did not. Needed so the
# rest of the app can check "is drug_provider available?" instead of
# needing to know HOW each service was started or why one might be missing.
@dataclass
class BootstrapResult:
    redis_cache: RedisProvider | None = None
    # redis_cache: the connected Redis provider, or None if Redis was unavailable.
    drug_provider: MilvusProvider | None = None
    # drug_provider: the connected Milvus client, or None if drug search
    # could not be started.
    guideline_agent: ClinicalGuidelineAgent | None = None
    # guideline_agent: the ready clinical-guideline agent, or None if file
    # search over guidelines could not be started.
    warnings: list[str] = field(default_factory=list)
    # warnings: human-readable messages about anything that was skipped.
    # "field(default_factory=list)" is needed instead of "= []" because a
    # mutable default like a list would otherwise be shared between every
    # BootstrapResult instance - a classic Python trap this avoids.

    async def close(self) -> None:
        # close: shuts down every service that WAS started, so network
        # connections are released cleanly when the app stops. Each check
        # below only closes a service if it was actually set up, since any
        # of them may be None.
        if self.redis_cache is not None:
            await self.redis_cache.disconnect()
        if self.drug_provider is not None:
            await self.drug_provider.disconnect()
        if self.guideline_agent is not None:
            await self.guideline_agent.close()


# bootstrap_services: the single function the rest of the app calls once,
# at startup, to get everything connected and ready. Needed so main.py (or
# wherever the app starts) does not need to know the details of connecting
# to Redis, Milvus, or the guidelines store itself.
async def bootstrap_services(
    *,
    use_drug_database: bool = True,
    use_clinical_guidelines: bool = True,
) -> BootstrapResult:
    """Connect to each service and keep only the ones that are ready.

    Docker services must already be running. A service that fails its check is left
    out of the result and recorded as a warning, so basic chat can still work.
    """
    # The "*" makes both arguments keyword-only, so callers must write
    # "bootstrap_services(use_drug_database=False)" rather than relying on
    # argument order, which keeps call sites self-explanatory.

    # Step 1: load settings and fail immediately if anything required for
    # the requested features is missing or invalid.
    settings = get_settings()
    check_settings(
        settings,
        use_drug_database=use_drug_database,
        use_clinical_guidelines=use_clinical_guidelines,
    )
    logger = get_component_logger("Bootstrap")
    result = BootstrapResult()
    # result: the BootstrapResult being built up step by step below, as
    # each service either succeeds or fails its check.

    # Step 2: Redis is always attempted (it backs session caching), but a
    # failure here does not stop the app - it just disables caching.
    redis_cache = RedisProvider(RedisCacheConfig.from_settings(settings))
    if await check_redis(redis_cache):
        result.redis_cache = redis_cache
    else:
        result.warnings.append("Redis is unavailable; session caching is disabled")

    # Step 3: drug search (Milvus) is only attempted if that feature flag
    # is on, since some deployments may not need it at all.
    if use_drug_database:
        drug_provider = MilvusProvider()
        if await check_milvus(drug_provider):
            result.drug_provider = drug_provider
        else:
            result.warnings.append("Drug search is unavailable; check Milvus and run ingestion")

    # Step 4: clinical guidelines (file search) is only attempted if that
    # feature flag is on.
    if use_clinical_guidelines:
        guideline_agent = ClinicalGuidelineAgent()
        if await prepare_guidelines_store(guideline_agent):
            result.guideline_agent = guideline_agent
        else:
            # If the store could not be made ready, close the agent we just
            # created instead of leaving it half-initialised and unused.
            await guideline_agent.close()
            result.warnings.append("Clinical guidelines are not ready; file search is disabled")

    # Step 5: log every warning collected above, so a developer watching
    # the logs can see exactly which features are degraded this run.
    for warning in result.warnings:
        logger.warning(warning)
    return result


# __all__ lists the names meant for other modules to import from this
# package's main path (e.g. ``from src.bootstrap import bootstrap_services``).
__all__ = ["BootstrapResult", "bootstrap_services"]
