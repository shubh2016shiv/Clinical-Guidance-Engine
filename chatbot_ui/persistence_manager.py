"""
Conversation Persistence Manager for Chainlit UI.

Handles JSON-based storage and retrieval of conversation history with correlation IDs
for request tracking and response correlation.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from chatbot_ui.config import CONVERSATION_DIR, MAX_HISTORY_MESSAGES


class ConversationPersistenceManager:
    """
    Manages conversation persistence using JSON file storage.

    Each conversation is stored as a separate JSON file with session ID as filename.
    Supports correlation between user requests and AI responses using request_id
    and response_id from the Response API.
    """

    def __init__(self, storage_dir: Optional[Path] = None):
        """
        Initialize the persistence manager.

        Args:
            storage_dir: Directory for storing conversation files.
                        Defaults to CONVERSATION_DIR from config.
        """
        self.storage_dir = storage_dir or CONVERSATION_DIR
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def _get_conversation_filepath(self, session_id: str) -> Path:
        """
        Get the file path for a conversation session.

        Args:
            session_id: Unique session identifier.

        Returns:
            Path to the conversation JSON file.
        """
        # Sanitize session_id to prevent directory traversal
        safe_session_id = "".join(
            c for c in session_id if c.isalnum() or c in ("-", "_")
        )
        return self.storage_dir / f"{safe_session_id}.json"

    def _create_new_conversation(self, session_id: str) -> Dict[str, Any]:
        """
        Create a new conversation structure.

        Args:
            session_id: Unique session identifier.

        Returns:
            New conversation dictionary.
        """
        return {
            "session_id": session_id,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "messages": [],
        }

    def load_conversation(self, session_id: str) -> Dict[str, Any]:
        """
        Load a conversation from storage.

        Args:
            session_id: Unique session identifier.

        Returns:
            Conversation dictionary, or new conversation if not found.
        """
        filepath = self._get_conversation_filepath(session_id)

        if not filepath.exists():
            return self._create_new_conversation(session_id)

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                conversation = json.load(f)
            return conversation
        except (json.JSONDecodeError, IOError) as e:
            # If file is corrupted, create new conversation
            print(f"Error loading conversation {session_id}: {e}")
            return self._create_new_conversation(session_id)

    def save_conversation(
        self,
        session_id: str,
        request_id: str,
        response_id: str,
        user_message: str,
        assistant_response: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Save a conversation turn (user message + assistant response).

        Args:
            session_id: Unique session identifier.
            request_id: Unique identifier for the user's request.
            response_id: Response ID from the Response API.
            user_message: The user's message text.
            assistant_response: The assistant's response text.
            metadata: Optional metadata (citations, tool usage, etc.).

        Returns:
            True if save successful, False otherwise.
        """
        try:
            conversation = self.load_conversation(session_id)

            timestamp = datetime.utcnow().isoformat()

            # Add user message
            conversation["messages"].append(
                {
                    "request_id": request_id,
                    "response_id": None,  # User messages don't have response IDs
                    "role": "user",
                    "content": user_message,
                    "timestamp": timestamp,
                    "metadata": {},
                }
            )

            # Add assistant message
            conversation["messages"].append(
                {
                    "request_id": request_id,  # Correlate with user request
                    "response_id": response_id,  # Response API response ID
                    "role": "assistant",
                    "content": assistant_response,
                    "timestamp": timestamp,
                    "metadata": metadata or {},
                }
            )

            # Update conversation metadata
            conversation["updated_at"] = timestamp

            # Enforce message limit
            if len(conversation["messages"]) > MAX_HISTORY_MESSAGES * 2:
                # Keep most recent messages (multiply by 2 for user+assistant pairs)
                conversation["messages"] = conversation["messages"][
                    -(MAX_HISTORY_MESSAGES * 2) :
                ]

            # Save to file
            filepath = self._get_conversation_filepath(session_id)
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(conversation, f, indent=2, ensure_ascii=False)

            return True

        except (IOError, json.JSONEncodeError) as e:
            print(f"Error saving conversation {session_id}: {e}")
            return False

    def load_conversation_history(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Load the message history for a conversation.

        Args:
            session_id: Unique session identifier.

        Returns:
            List of message dictionaries in chronological order.
        """
        conversation = self.load_conversation(session_id)
        return conversation.get("messages", [])

    def get_conversation_summary(self, session_id: str) -> Dict[str, Any]:
        """
        Get a summary of the conversation metadata.

        Args:
            session_id: Unique session identifier.

        Returns:
            Dictionary with conversation summary information.
        """
        conversation = self.load_conversation(session_id)

        messages = conversation.get("messages", [])
        user_messages = [m for m in messages if m.get("role") == "user"]
        assistant_messages = [m for m in messages if m.get("role") == "assistant"]

        return {
            "session_id": session_id,
            "created_at": conversation.get("created_at"),
            "updated_at": conversation.get("updated_at"),
            "total_messages": len(messages),
            "user_messages": len(user_messages),
            "assistant_messages": len(assistant_messages),
            "has_citations": any(
                m.get("metadata", {}).get("citations") for m in assistant_messages
            ),
        }

    def delete_conversation(self, session_id: str) -> bool:
        """
        Delete a conversation from storage.

        Args:
            session_id: Unique session identifier.

        Returns:
            True if deletion successful, False otherwise.
        """
        filepath = self._get_conversation_filepath(session_id)

        if not filepath.exists():
            return True

        try:
            filepath.unlink()
            return True
        except IOError as e:
            print(f"Error deleting conversation {session_id}: {e}")
            return False

    def list_conversations(self) -> List[Dict[str, Any]]:
        """
        List all stored conversations with summary information.

        Returns:
            List of conversation summary dictionaries.
        """
        summaries = []

        for filepath in self.storage_dir.glob("*.json"):
            try:
                session_id = filepath.stem
                summary = self.get_conversation_summary(session_id)
                summaries.append(summary)
            except Exception as e:
                print(f"Error reading conversation {filepath}: {e}")
                continue

        # Sort by updated_at descending (most recent first)
        summaries.sort(key=lambda x: x.get("updated_at", ""), reverse=True)

        return summaries

    def cleanup_old_conversations(self, days: int = 30) -> int:
        """
        Delete conversations older than specified days.

        Args:
            days: Number of days to keep conversations.

        Returns:
            Number of conversations deleted.
        """
        from datetime import timedelta

        cutoff_date = datetime.utcnow() - timedelta(days=days)
        deleted_count = 0

        for filepath in self.storage_dir.glob("*.json"):
            try:
                session_id = filepath.stem
                conversation = self.load_conversation(session_id)

                updated_at_str = conversation.get("updated_at")
                if updated_at_str:
                    updated_at = datetime.fromisoformat(updated_at_str)

                    if updated_at < cutoff_date:
                        if self.delete_conversation(session_id):
                            deleted_count += 1
            except Exception as e:
                print(f"Error processing conversation {filepath}: {e}")
                continue

        return deleted_count
