"""Keep the command-line entry point wired to the current agents."""

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from src.app import answer_question
from src.drugs_agent import DrugAnswer, DrugMatch, format_drug_answer_as_markdown_list


class AnswerQuestionTests(unittest.IsolatedAsyncioTestCase):
    async def test_guideline_question_uses_guideline_agent_and_closes_services(self) -> None:
        guideline_agent = MagicMock()
        guideline_agent.answer_guideline_question = AsyncMock(
            return_value=SimpleNamespace(text="Guideline answer")
        )
        startup = SimpleNamespace(guideline_agent=guideline_agent, close=AsyncMock())

        with patch("src.app.bootstrap_services", AsyncMock(return_value=startup)) as bootstrap:
            result = await answer_question("When should this be used?", source="guidelines")

        self.assertEqual(result, "Guideline answer")
        bootstrap.assert_awaited_once_with(use_drug_database=False, use_clinical_guidelines=True)
        guideline_agent.answer_guideline_question.assert_awaited_once_with(
            "When should this be used?"
        )
        startup.close.assert_awaited_once()

    async def test_drug_question_uses_chat_agent_and_closes_clients(self) -> None:
        startup = SimpleNamespace(drug_provider=MagicMock(), close=AsyncMock())
        drug_llm_provider = MagicMock(close=AsyncMock())
        drug_agent = MagicMock()
        drug_match = DrugMatch(
            drug_name="Metformin",
            drug_class="antidiabetic agents",
            drug_sub_class=None,
            therapeutic_category=None,
            route_of_administration="oral",
            formulation="tablet",
            dosage_strengths=["500 mg"],
        )
        drug_answer = DrugAnswer(found=True, matches=[drug_match])
        drug_agent.answer_drug_search_query = AsyncMock(return_value=drug_answer)

        with (
            patch("src.app.bootstrap_services", AsyncMock(return_value=startup)) as bootstrap,
            patch("src.app.OpenAIProvider", return_value=drug_llm_provider) as provider_class,
            patch("src.app.DrugSearcher") as searcher_class,
            patch("src.app.DrugAgent", return_value=drug_agent) as agent_class,
        ):
            result = await answer_question("metformin", source="drugs")

        # The layout itself is covered by test_drug_formatter.py; here we only check
        # that the CLI returns the formatter's rendering of the agent's answer.
        self.assertEqual(result, format_drug_answer_as_markdown_list(drug_answer))
        self.assertIn("**Metformin**", result)
        self.assertIn("500 mg", result)
        bootstrap.assert_awaited_once_with(use_drug_database=True, use_clinical_guidelines=False)
        provider_class.assert_called_once()
        searcher_class.assert_called_once_with(startup.drug_provider)
        agent_class.assert_called_once_with(searcher_class.return_value, drug_llm_provider)
        drug_agent.answer_drug_search_query.assert_awaited_once_with("metformin")
        drug_llm_provider.close.assert_awaited_once()
        startup.close.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
