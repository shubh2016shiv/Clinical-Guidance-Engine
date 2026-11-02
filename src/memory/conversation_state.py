"""Declare the message-history field and its LangGraph merge behavior."""

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


# ---------------------------------------------------------------------------
# MEMORY STATE — Keep the conversation when nodes return individual new messages
# ---------------------------------------------------------------------------
# New message IDs append; matching IDs update existing messages. A graph extends
# this state with its own working fields, which use replacement unless annotated.
# Official reference:
# https://docs.langchain.com/oss/python/langgraph/graph-api#working-with-messages-in-graph-state
class ConversationHistoryState(TypedDict, total=False):
    """Optional checkpointed history shared by graph-specific state definitions."""

    conversation_messages: Annotated[list[AnyMessage], add_messages]
