"""Node (Path B, step 4): write one final answer from the guidelines and the drugs found.

This is the answer the user sees on Path B. The model's text is streamed to the
user while it is written, and the list of drugs is added at the end.
"""

import json
from typing import Any

from langgraph.config import get_stream_writer

from src.drugs_agent import format_drug_answer_as_markdown_list
from src.prompts.orchestrator_system_prompts import ANSWER_CONSOLIDATION_SYSTEM_PROMPT

from ..conversation_history import select_recent_previous_messages_for_chat_model
from ..schema import OrchestratorState
from ..structured_model_call import TEMPERATURE


def make_stream_combined_answer_node(chat_completions_provider: Any, model_name: str):
    """Build the node. The graph calls the returned function with its shared state."""

    async def stream_combined_answer(state: OrchestratorState) -> dict[str, Any]:
        """Have the model write one final answer, stream it, then append the drug list."""
        send_text_to_user = get_stream_writer()
        drug_answer = state["drug_answer"]

        # Everything the model needs to write the final answer.
        model_input = json.dumps(
            {
                "question": state["clinical_question"],
                "guideline_answer": state["cited_guideline_text"],
                "searched_drug_classes": state["drug_classes"],
                "drug_records": [
                    drug.model_dump(exclude_none=True) for drug in drug_answer.matches
                ],
            }
        )

        text_stream = await chat_completions_provider.execute_chat_completion(
            [
                {"role": "system", "content": ANSWER_CONSOLIDATION_SYSTEM_PROMPT},
                # Earlier exchanges sit between the system prompt and the current
                # question, so the model can keep its answer consistent with them.
                *select_recent_previous_messages_for_chat_model(state["conversation_messages"]),
                {"role": "user", "content": model_input},
            ],
            stream=True,
            model=model_name,
            temperature=TEMPERATURE,
        )

        # Forward the model's text to the user piece by piece.
        async for text_piece in text_stream:
            send_text_to_user(text_piece)

        # Finally, add the list of drugs under the answer.
        send_text_to_user("\n\n" + format_drug_answer_as_markdown_list(drug_answer))

        return {}  # nothing new to save; the text was already streamed out

    return stream_combined_answer
