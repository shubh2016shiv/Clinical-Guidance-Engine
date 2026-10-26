"""Node (Path B, step 3): look up every recommended drug class in the catalog."""

import asyncio
from typing import Any

from src.drugs_agent import DrugAnswer, DrugMatch

from ..schema import OrchestratorState


def make_search_drugs_node(drug_agent: Any):
    """Build the node. The graph calls the returned function with its shared state."""

    async def search_drugs(state: OrchestratorState) -> dict[str, Any]:
        """Search the drug catalog for every class, all at the same time."""
        # asyncio.gather runs all the searches in parallel and waits for all of them.
        results_per_class = await asyncio.gather(
            *(drug_agent.answer_drug_class_query(name) for name in state["drug_classes"])
        )

        # The same drug can show up under several classes. Keep only the first copy
        # of each drug name.
        drugs_by_name: dict[str, DrugMatch] = {}
        for result in results_per_class:
            for drug in result.matches:
                drugs_by_name.setdefault(drug.drug_name, drug)

        all_drugs = list(drugs_by_name.values())
        return {"drug_answer": DrugAnswer(found=bool(all_drugs), matches=all_drugs)}

    return search_drugs
