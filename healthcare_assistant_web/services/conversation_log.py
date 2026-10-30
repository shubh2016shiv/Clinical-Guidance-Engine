"""Saves each question and answer to a JSON file, one file per browser tab."""

import json
from datetime import UTC, datetime
from pathlib import Path

from src.orchestrator import OrchestratorAnswer

from ..web_ui_settings import CONVERSATIONS_DIR


class ConversationLog:
    """Keeps a record of every turn of a conversation on disk."""

    def __init__(self, directory: Path = CONVERSATIONS_DIR) -> None:
        self.directory = directory

    def save_turn(
        self, session_id: str, request_id: str, question: str, answer: OrchestratorAnswer
    ) -> None:
        """Add one question and its answer to this tab's conversation file."""
        self.directory.mkdir(parents=True, exist_ok=True)
        conversation_file = self.directory / f"{session_id}.json"

        if conversation_file.exists():
            conversation = json.loads(conversation_file.read_text(encoding="utf-8"))
        else:
            conversation = {"session_id": session_id, "turns": []}

        conversation["turns"].append(
            {
                "request_id": request_id,
                "time": datetime.now(UTC).isoformat(),
                "question": question,
                "answer": answer.text,
                "route": answer.route,
                "sources": [source.filename for source in answer.sources],
                "drug_classes": answer.drug_classes,
            }
        )
        conversation_file.write_text(json.dumps(conversation, indent=2), encoding="utf-8")
