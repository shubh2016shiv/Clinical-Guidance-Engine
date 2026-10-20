"""Ensure the hosted clinical guidelines store is ready for file search."""

from src.logs import get_component_logger
from src.response_api_agent.asclepius_agent import AsclepiusHealthcareAgent

logger = get_component_logger("GuidelinesStartup")


async def prepare_guidelines_store(agent: AsclepiusHealthcareAgent) -> str | None:
    """Reuse or create the store and require all attached files to be ready."""
    store_id = await agent.initialize_knowledge_base()
    if not store_id:
        return None

    try:
        store_manager = agent.response_manager.vector_store_manager
        store = await store_manager.get_vector_store(store_id)
        files = await store_manager.get_vector_store_files(store_id) if store else []
        if (
            store
            and store.get("status") == "completed"
            and files
            and all(file.get("status") == "completed" for file in files)
        ):
            return store_id
    except Exception as exc:
        logger.warning("Could not verify guidelines store", error=str(exc))

    agent._vector_store_id = None
    return None
