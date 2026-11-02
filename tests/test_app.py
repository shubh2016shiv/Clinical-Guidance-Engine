"""Keep the command-line entry point wired to the current agents."""

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from src.app import answer_from_guidelines


class AnswerFromGuidelinesTests(unittest.IsolatedAsyncioTestCase):
    async def test_guideline_question_uses_guideline_agent_and_closes_services(self) -> None:
        guideline_agent = MagicMock()
        guideline_agent.answer_guideline_question = AsyncMock(
            return_value=SimpleNamespace(text="Guideline answer")
        )
        startup = SimpleNamespace(guideline_agent=guideline_agent, close=AsyncMock())

        with patch("src.app.bootstrap_services", AsyncMock(return_value=startup)) as bootstrap:
            result = await answer_from_guidelines("When should this be used?")

        self.assertEqual(result, "Guideline answer")
        bootstrap.assert_awaited_once_with(use_drug_database=False, use_clinical_guidelines=True)
        guideline_agent.answer_guideline_question.assert_awaited_once_with(
            "When should this be used?"
        )
        startup.close.assert_awaited_once()

    async def test_unavailable_guidelines_raise_and_still_close_services(self) -> None:
        startup = SimpleNamespace(guideline_agent=None, close=AsyncMock())

        with patch("src.app.bootstrap_services", AsyncMock(return_value=startup)):
            with self.assertRaises(RuntimeError):
                await answer_from_guidelines("When should this be used?")

        startup.close.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
