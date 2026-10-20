"""Command-line demo for drug search or clinical guideline questions."""

import argparse
import asyncio

from src.adapters.llm_provider import APIType, OpenAIProvider
from src.bootstrap import bootstrap_services
from src.drugs_agent import DrugAgent, DrugSearcher


async def answer_question(question: str, *, source: str) -> str:
    """Check the needed services, answer one question, and close owned clients."""
    startup = await bootstrap_services(
        use_drug_database=source == "drugs",
        use_clinical_guidelines=source == "guidelines",
    )
    drug_llm_provider = None
    try:
        if source == "guidelines":
            if startup.guideline_agent is None:
                raise RuntimeError(
                    "Clinical guidelines are unavailable. Check the startup warnings."
                )
            answer = await startup.guideline_agent.answer_guideline_question(question)
            return answer.text

        if startup.drug_provider is None:
            raise RuntimeError("Drug search is unavailable. Check Milvus and run ingestion.")
        drug_llm_provider = OpenAIProvider(api_type=APIType.CHAT_COMPLETION)
        drug_agent = DrugAgent(DrugSearcher(startup.drug_provider), drug_llm_provider)
        answer = await drug_agent.answer_drug_search_query(question)
        return answer.model_dump_json(indent=2)
    finally:
        if drug_llm_provider is not None:
            await drug_llm_provider.close()
        await startup.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", help="Question to answer")
    parser.add_argument("--source", choices=("drugs", "guidelines"), default="guidelines")
    args = parser.parse_args()
    print(asyncio.run(answer_question(args.question, source=args.source)))


if __name__ == "__main__":
    main()
