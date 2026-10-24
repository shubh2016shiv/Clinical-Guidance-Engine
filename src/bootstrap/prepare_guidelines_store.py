"""Ensure the hosted clinical guidelines store is ready for file search.

Big-picture flow:
1. Ask the clinical guideline agent to reuse or create its file-search store
   on OpenAI (uploading guideline documents if needed).
2. If that fails for any expected reason, log a warning and report "not
   ready" (None) instead of crashing the whole app - citations just will
   not be available this session.
"""

from openai import OpenAIError

# OpenAIError is the base class for every error the OpenAI SDK can raise
# (bad request, network failure, rate limit, etc.). Catching it here means
# "any kind of OpenAI-side failure", without needing to list each one.
from src.clinical_guideline_agent import ClinicalGuidelineAgent
from src.logs import get_component_logger

logger = get_component_logger("GuidelinesStartup")
# A logger lets this file record a warning when the guidelines feature
# cannot start, without needing to crash the app to reveal why.


# prepare_guidelines_store: the single entry point bootstrap uses to turn
# clinical-guideline file search on. Needed so start-up failures here (for
# example, missing guideline files, or an OpenAI outage) are handled the
# same gentle way as the Redis/Milvus checks in check_redis_and_milvus.py.
async def prepare_guidelines_store(agent: ClinicalGuidelineAgent) -> str | None:
    """Reuse or create the store, returning its ID or None if it cannot be made ready."""
    # The return type "str | None" means: either the store's ID (success),
    # or None (it could not be made ready). The caller in __init__.py only
    # checks "if this is truthy", so the ID itself does not need to be used
    # there - the agent object keeps its own reference to the store.
    try:
        # Step 1: ask the agent to do the actual work - checking for an
        # existing store, or creating and populating a new one.
        return await agent.prepare_guidelines_store()
    except (ValueError, RuntimeError, OpenAIError) as exc:
        # Step 2: catch several DIFFERENT kinds of expected failure at once
        # by listing them in a tuple - a bad guidelines folder (ValueError),
        # an internal processing problem (RuntimeError), or an OpenAI-side
        # failure (OpenAIError). Any other, unexpected exception type is
        # deliberately NOT caught here, so it still surfaces as a real bug.
        logger.warning("Clinical guidelines are not ready", error=str(exc))
        return None
