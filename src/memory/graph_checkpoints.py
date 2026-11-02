"""Create LangGraph checkpoint storage and select a conversation's saved state."""

from uuid import uuid4

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

# ---------------------------------------------------------------------------
# MEMORY SETUP — Restore the same agent-result types already stored in checkpoints
# ---------------------------------------------------------------------------
# These are serialization paths, not agent imports or workflow decisions.
# Preserve them when moving orchestration code so existing saved data stays readable.
CHECKPOINT_SAVED_CLASSES = [
    ("src.clinical_guideline_agent.schema", "GuidelineAnswer"),
    ("src.clinical_guideline_agent.schema", "FileCitation"),
    ("src.drugs_agent.schema", "DrugAnswer"),
    ("src.drugs_agent.schema", "DrugMatch"),
]


def create_in_memory_checkpointer() -> InMemorySaver:
    """Return a fresh process-local saver with the application's result serializer.

    Checkpoints survive repeated calls using this saver, not process restarts.
    A caller can still supply a different checkpointer to the orchestrator.
    Official reference: https://docs.langchain.com/oss/python/langgraph/persistence
    """
    return InMemorySaver(serde=JsonPlusSerializer(allowed_msgpack_modules=CHECKPOINT_SAVED_CLASSES))


# ---------------------------------------------------------------------------
# MEMORY IDENTITY — Reuse an explicit conversation, otherwise start a fresh one
# ---------------------------------------------------------------------------
def create_conversation_run_config(conversation_thread_id: str | None = None) -> RunnableConfig:
    """Select graph checkpoint history without changing any agent's separate memory.

    Preserve the existing behavior: None or an empty ID creates a fresh UUID.
    Official reference: https://docs.langchain.com/oss/python/langgraph/checkpointers#threads
    """
    return {"configurable": {"thread_id": conversation_thread_id or str(uuid4())}}
