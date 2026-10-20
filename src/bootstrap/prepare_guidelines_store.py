"""Ensure the hosted clinical guidelines store is ready for file search."""

from openai import OpenAIError

from src.clinical_guideline_agent import ClinicalGuidelineAgent
from src.logs import get_component_logger

logger = get_component_logger("GuidelinesStartup")


async def prepare_guidelines_store(agent: ClinicalGuidelineAgent) -> str | None:
    """Reuse or create the store, returning its ID or None if it cannot be made ready."""
    try:
        return await agent.prepare_guidelines_store()
    except (ValueError, RuntimeError, OpenAIError) as exc:
        logger.warning("Clinical guidelines are not ready", error=str(exc))
        return None
