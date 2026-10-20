"""Check that ChatManager session IDs stay unique and keep their format."""

import re
import unittest
from unittest.mock import patch

from src.response_api_agent.managers.chat_manager import ChatManager


class SessionIdTests(unittest.TestCase):
    def test_ids_have_prefix_timestamp_and_random_suffix(self) -> None:
        with patch("time.time", return_value=1730804400.9):
            session_id = ChatManager.generate_session_id()

        self.assertRegex(session_id, r"^session_1730804400_[0-9a-f]{8}$")

    def test_ids_generated_in_the_same_second_are_unique(self) -> None:
        with patch("time.time", return_value=1730804400.0):
            session_ids = {ChatManager.generate_session_id() for _ in range(5000)}

        self.assertEqual(len(session_ids), 5000)

    def test_every_id_passes_the_session_prefix_check_used_by_the_managers(self) -> None:
        session_id = ChatManager.generate_session_id()

        self.assertTrue(session_id.startswith("session_"))
        self.assertIsNone(re.search(r"\s", session_id))


if __name__ == "__main__":
    unittest.main()
