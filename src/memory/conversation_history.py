"""Select recent history for triage without changing the stored conversation.

LangGraph's checkpointer stores the full history by thread ID. This module only
selects the previous messages sent to one model call; it does not store or trim history.
"""

from langchain_core.messages import AnyMessage

# One "exchange" is one user message plus the assistant reply to it.
# ---------------------------------------------------------------------------
# HISTORY SELECTION — Limit model input without deleting saved history
# ---------------------------------------------------------------------------
# The ten-exchange window is our policy. Checkpoint history stays complete.
# Official reference: https://docs.langchain.com/oss/python/langgraph/checkpointers
MAXIMUM_PREVIOUS_EXCHANGES_SENT_TO_CHAT_MODEL = 10
MAXIMUM_PREVIOUS_MESSAGES_SENT_TO_CHAT_MODEL = MAXIMUM_PREVIOUS_EXCHANGES_SENT_TO_CHAT_MODEL * 2

# Translate stored message types to chat-model roles.
CHAT_COMPLETIONS_ROLE_BY_LANGCHAIN_MESSAGE_TYPE = {
    "human": "user",
    "ai": "assistant",
}


def select_recent_previous_messages_for_chat_model(
    conversation_messages: list[AnyMessage],
) -> list[dict[str, str]]:
    """Return recent previous messages as role/content dictionaries, oldest first.

    Input contains user/assistant messages with the current question LAST. Exclude
    that question because the caller sends it separately. Take up to 20 previous
    messages (10 complete exchanges), then drop any leading assistant replies whose
    questions fell outside the window. Empty input returns an empty list.
    The stored list and its messages are not modified.
    """
    # ---------------------------------------------------------------------------
    # HISTORY 1 — Exclude the current question; the caller sends it separately
    # ---------------------------------------------------------------------------
    previous_messages = conversation_messages[:-1]

    # ---------------------------------------------------------------------------
    # HISTORY 2 — Take the recent window and drop an orphan assistant prefix
    # ---------------------------------------------------------------------------
    # Failed turns can leave consecutive questions. Select messages, not assumed pairs.
    recent_messages = previous_messages[-MAXIMUM_PREVIOUS_MESSAGES_SENT_TO_CHAT_MODEL:]

    # Never begin with an assistant reply whose question was cut off.
    while recent_messages and recent_messages[0].type != "human":
        recent_messages = recent_messages[1:]

    # ---------------------------------------------------------------------------
    # HISTORY 3 — Translate roles into new model-input dictionaries
    # ---------------------------------------------------------------------------
    # Selection does not edit the saved message objects.
    return [
        {
            "role": CHAT_COMPLETIONS_ROLE_BY_LANGCHAIN_MESSAGE_TYPE[message.type],
            "content": message.content,
        }
        for message in recent_messages
    ]
