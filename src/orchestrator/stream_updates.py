"""Shared graph event bridge; node callbacks use the active custom-stream writer.

Keeping this bridge outside graph.py lets nodes emit events without importing
the graph builder. Work functions still receive an explicit text callback.
Official reference: https://docs.langchain.com/oss/python/langgraph/streaming
"""

from langgraph.config import get_stream_writer


# ---------------------------------------------------------------------------
# STREAM BRIDGE — Emit display events separately from state updates
# ---------------------------------------------------------------------------
# These helpers require an active graph run. Work functions receive the text helper
# as an explicit callback, so those functions also work outside LangGraph.
# Official reference: https://docs.langchain.com/oss/python/langgraph/streaming
def send_answer_text(text: str) -> None:
    """Send a piece of the answer itself, to be shown as it arrives."""
    get_stream_writer()({"kind": "text", "text": text})


def send_status(message: str) -> None:
    """Say what is being worked on. Never part of the answer, and never saved."""
    get_stream_writer()({"kind": "status", "text": message})
