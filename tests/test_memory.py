"""Check LangGraph memory policy independently of workflow and model calls."""

import unittest
from uuid import UUID

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph

from src.memory import (
    ConversationHistoryState,
    create_conversation_run_config,
    create_in_memory_checkpointer,
    select_recent_previous_messages_for_chat_model,
)


class GraphCheckpointTests(unittest.IsolatedAsyncioTestCase):
    def test_config_reuses_an_explicit_id_and_generates_fresh_ids_when_omitted(self) -> None:
        self.assertEqual(
            create_conversation_run_config("existing-thread"),
            {"configurable": {"thread_id": "existing-thread"}},
        )
        generated_ids = [
            create_conversation_run_config(value)["configurable"]["thread_id"]
            for value in (None, None, "")
        ]
        self.assertEqual(len(set(generated_ids)), 3)
        for thread_id in generated_ids:
            self.assertEqual(str(UUID(thread_id)), thread_id)

    async def test_history_appends_updates_by_id_and_stays_isolated_between_threads(self) -> None:
        def retain_history(state: ConversationHistoryState) -> dict:
            # Input messages are merged by the reducer; the node needs no extra work.
            return {}

        graph_builder = StateGraph(ConversationHistoryState)
        graph_builder.add_node(retain_history)
        graph_builder.add_edge(START, "retain_history")
        graph_builder.add_edge("retain_history", END)
        graph = graph_builder.compile(checkpointer=create_in_memory_checkpointer())
        first_thread = create_conversation_run_config("first")

        await graph.ainvoke(
            {"conversation_messages": [HumanMessage(content="question", id="question-1")]},
            first_thread,
        )
        await graph.ainvoke(
            {"conversation_messages": [AIMessage(content="answer", id="answer-1")]},
            first_thread,
        )
        await graph.ainvoke(
            {"conversation_messages": [HumanMessage(content="edited question", id="question-1")]},
            first_thread,
        )
        second_thread = create_conversation_run_config("second")
        await graph.ainvoke(
            {"conversation_messages": [HumanMessage(content="separate question")]}, second_thread
        )

        first_snapshot = await graph.aget_state(first_thread)
        second_snapshot = await graph.aget_state(second_thread)
        self.assertEqual(
            [message.content for message in first_snapshot.values["conversation_messages"]],
            ["edited question", "answer"],
        )
        self.assertEqual(
            [message.content for message in second_snapshot.values["conversation_messages"]],
            ["separate question"],
        )


class HistorySelectionTests(unittest.TestCase):
    """Select model history without changing the conversation stored by the graph."""

    def test_empty_first_and_completed_turns_exclude_the_current_question(self) -> None:
        cases = (
            ([], []),
            ([HumanMessage(content="current")], []),
            (
                [
                    HumanMessage(content="earlier"),
                    AIMessage(content="reply"),
                    HumanMessage(content="current"),
                ],
                [{"role": "user", "content": "earlier"}, {"role": "assistant", "content": "reply"}],
            ),
        )
        for messages, expected in cases:
            with self.subTest(message_count=len(messages)):
                original_messages = [message.model_dump() for message in messages]

                selected = select_recent_previous_messages_for_chat_model(messages)

                self.assertEqual(selected, expected)
                self.assertEqual([message.model_dump() for message in messages], original_messages)
                if selected:
                    selected[0]["content"] = "changed model input"
                    self.assertEqual(
                        [message.model_dump() for message in messages], original_messages
                    )

    def test_window_drops_an_orphan_reply_and_retains_an_unanswered_question(self) -> None:
        messages = []
        for number in range(11):
            messages.extend(
                [HumanMessage(content=f"question {number}"), AIMessage(content=f"reply {number}")]
            )
        messages.extend([HumanMessage(content="unanswered"), HumanMessage(content="current")])
        original_messages = [message.model_dump() for message in messages]

        selected = select_recent_previous_messages_for_chat_model(messages)

        # The 20-message slice starts with reply 1. Its question is outside the
        # window, so remove that reply while preserving the later unanswered turn.
        expected = []
        for number in range(2, 11):
            expected.extend(
                [
                    {"role": "user", "content": f"question {number}"},
                    {"role": "assistant", "content": f"reply {number}"},
                ]
            )
        expected.append({"role": "user", "content": "unanswered"})
        self.assertEqual(selected, expected)
        self.assertEqual([message.model_dump() for message in messages], original_messages)
