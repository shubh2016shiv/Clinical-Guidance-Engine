"""How earlier turns of a conversation are remembered and handed to the chat model.

There are two separate jobs, and keeping them apart is the key idea:

1. REMEMBERING (storage).
   The graph state has a list called `conversation_messages`. A LangGraph
   "checkpointer" saves that state after every run under a `thread_id` (one id per
   conversation). On the next question with the same `thread_id`, LangGraph loads the
   saved list back, and the `add_messages` reducer appends the new message to it.
   So the stored list keeps growing: the FULL history is always kept.

2. SELECTING (what the model sees).
   A chat model has a limited context window, so we do not send the full history.
   Right before calling the model we pick only the most recent messages
   (`select_recent_previous_messages_for_chat_model` below). Selecting never changes
   the stored list; it only decides what one model call receives.
"""

from langchain_core.messages import AnyMessage, trim_messages

# One "exchange" is one user message plus the assistant reply to it.
MAXIMUM_PREVIOUS_EXCHANGES_SENT_TO_CHAT_MODEL = 10
MAXIMUM_PREVIOUS_MESSAGES_SENT_TO_CHAT_MODEL = MAXIMUM_PREVIOUS_EXCHANGES_SENT_TO_CHAT_MODEL * 2

# LangChain names the speakers "human" and "ai"; the chat completions API calls
# them "user" and "assistant". This table translates between the two.
CHAT_COMPLETIONS_ROLE_BY_LANGCHAIN_MESSAGE_TYPE = {
    "human": "user",
    "ai": "assistant",
}


def select_recent_previous_messages_for_chat_model(
    conversation_messages: list[AnyMessage],
) -> list[dict[str, str]]:
    """Return up to the last 10 exchanges BEFORE the current question, ready to send.

    `conversation_messages` is the stored history. By the time a node runs, the
    current question has already been appended as the LAST message, so we drop it
    here: the node sends the current question itself, in its own format.

    The result is a list like [{"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}, ...], oldest first.
    """
    previous_messages = conversation_messages[:-1]

    recent_previous_messages = trim_messages(
        previous_messages,
        strategy="last",  # when there are too many, drop the OLDEST ones
        max_tokens=MAXIMUM_PREVIOUS_MESSAGES_SENT_TO_CHAT_MODEL,
        token_counter=len,  # count each message as 1, so "max_tokens" means "max messages"
        start_on="human",  # never begin with an assistant reply that has lost its question
        include_system=False,  # history holds no system prompt; each node adds its own
    )

    return [
        {
            "role": CHAT_COMPLETIONS_ROLE_BY_LANGCHAIN_MESSAGE_TYPE[message.type],
            "content": message.content,
        }
        for message in recent_previous_messages
    ]
