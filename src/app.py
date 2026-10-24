"""Command-line demo for drug search or clinical guideline questions."""

import argparse
import asyncio
import sys
from collections.abc import AsyncIterator

from src.adapters.llm_provider import APIType, OpenAIProvider
from src.bootstrap import bootstrap_services
from src.clinical_guideline_agent.citations.inline_citations import citation_marker
from src.clinical_guideline_agent.schema import FileCitation
from src.drugs_agent import DrugAgent, DrugSearcher, format_drug_answer_as_markdown_table
from src.orchestrator import ClinicalDrugOrchestrator, OrchestratorUpdate


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
        return format_drug_answer_as_markdown_table(answer)
    finally:
        if drug_llm_provider is not None:
            await drug_llm_provider.close()
        await startup.close()


async def stream_orchestrated_answer(question: str) -> AsyncIterator[OrchestratorUpdate]:
    """Let the orchestrator choose guidelines alone or guidelines plus drug data."""
    startup = await bootstrap_services()
    chat_completions_provider = OpenAIProvider(api_type=APIType.CHAT_COMPLETION)
    try:
        if startup.guideline_agent is None or startup.drug_provider is None:
            raise RuntimeError(
                "The orchestrator needs clinical guidelines and drug search. "
                f"Startup warnings: {startup.warnings}"
            )
        drug_agent = DrugAgent(DrugSearcher(startup.drug_provider), chat_completions_provider)
        orchestrator = ClinicalDrugOrchestrator(
            startup.guideline_agent, drug_agent, chat_completions_provider
        )
        async for orchestrator_update in orchestrator.stream_clinical_drug_answer_updates(question):
            yield orchestrator_update
    finally:
        await chat_completions_provider.close()
        await startup.close()


async def _print_streamed_answer(question: str) -> None:
    async for orchestrator_update in stream_orchestrated_answer(question):
        if orchestrator_update.type == "text":
            print(orchestrator_update.text, end="", flush=True)
        elif orchestrator_update.answer is not None:
            print(_references_block(orchestrator_update.answer.sources))


def _references_block(sources: list[FileCitation]) -> str:
    """Number the sources like the inline markers; clinical-only answers stream unmarked."""
    if not sources:
        return ""
    numbered_sources = (
        f"{citation_marker(source_number)} {source.filename}"
        for source_number, source in enumerate(sources, start=1)
    )
    return "\n\nReferences:\n" + "\n".join(numbered_sources)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", help="Question to answer")
    parser.add_argument("--source", choices=("auto", "drugs", "guidelines"), default="auto")
    args = parser.parse_args()
    # Answers contain symbols such as "≥" that the Windows code page cannot print.
    sys.stdout.reconfigure(encoding="utf-8")
    if args.source == "auto":
        asyncio.run(_print_streamed_answer(args.question))
    else:
        print(asyncio.run(answer_question(args.question, source=args.source)))


if __name__ == "__main__":
    main()
