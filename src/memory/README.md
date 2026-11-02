# LangGraph conversation memory

This package owns graph checkpoint setup, conversation thread identity, message merging
and selection of prior messages for model input. It has no imports from the orchestrator
or agent implementations. It does not route questions, search drugs or generate answers.

| File | Responsibility |
| --- | --- |
| `graph_checkpoints.py` | Create the default in-memory saver, configure result serialization and select thread IDs |
| `conversation_state.py` | Declare `conversation_messages` and its `add_messages` reducer |
| `conversation_history.py` | Select previous model-input messages without truncating stored history |
| `__init__.py` | Export memory capabilities for graph callers |

## How a graph uses this package

1. Extend `ConversationHistoryState` with workflow-specific fields. The history field
   retains its existing name, `conversation_messages`; new message IDs append and
   matching IDs update the stored message.
2. Create a saver with `create_in_memory_checkpointer()`, or supply another supported
   checkpointer. The default saver preserves the existing agent-result serializer paths
   and loses its contents when the process ends.
3. Call `create_conversation_run_config(thread_id)` to select saved graph history.
   A missing or empty ID creates a fresh UUID. Reuse both the saver and ID to continue.
4. Supply new messages in graph state updates. Graph nodes still decide when a user
   question or completed assistant answer is ready to enter the conversation.
5. Use `select_recent_previous_messages_for_chat_model()` when preparing model input.
   Its input includes the current question last. It excludes that question, takes up
   to 20 previous messages and removes an orphan assistant prefix. Full history remains
   in the checkpoint; this selection does not edit saved messages.

The orchestrator retains its question-result fields, graph wiring, message-producing
nodes, and graph invocation/final-result retrieval. It consumes this package's memory
capabilities instead of implementing storage and history policy itself.

The clinical guideline agent's separate Responses API response chain belongs to its
instance. It is not LangGraph memory, and a graph thread ID does not reset that chain.
Redis-backed chat/session designs and UI conversation logs are also outside this package.

## Official references

- [State and message reducers](https://docs.langchain.com/oss/python/langgraph/graph-api#working-with-messages-in-graph-state)
- [Checkpoint storage and memory lifetime](https://docs.langchain.com/oss/python/langgraph/persistence)
- [Threads and snapshots](https://docs.langchain.com/oss/python/langgraph/checkpointers)

## Verification

Run `python -m unittest tests.test_memory` for independent memory tests. They exercise
history selection, generated and explicit IDs, message updates by ID and thread isolation
without orchestrator or agent calls. The orchestrator suite additionally exercises typed
agent-result restoration and conversation continuity through complete workflow runs.
