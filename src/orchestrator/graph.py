"""Wire the four numbered nodes into the clinical and catalog-only paths.

Read the node files in nodes/ for state updates and streaming callbacks.
Node 02 forwards guideline updates directly; the clinical agent owns retrieval and
interpretation. See README.md for the reading order and branch map.
"""

from typing import Any

from langgraph.graph import END, START, StateGraph

from .nodes import (
    finalize_answer,
    search_catalog,
    stream_guideline_answer,
    triage,
)
from .schema import OrchestratorContext, OrchestratorState


# ---------------------------------------------------------------------------
# WORKFLOW MAP — Read the path before the individual steps
# ---------------------------------------------------------------------------
# Nodes return updates to shared state; edges choose the next node.
# Our route strings and four node names are application contracts.
# Official reference: https://docs.langchain.com/oss/python/langgraph/graph-api
def build_orchestrator_graph(checkpointer: Any) -> Any:
    """Compile the clinical and catalog-only paths with the supplied checkpointer."""
    # ---------------------------------------------------------------------------
    # BUILD 1 — Declare saved state and dependencies supplied for each run
    # ---------------------------------------------------------------------------
    # State holds working data. runtime.context holds clients and model selection.
    # Official reference: https://docs.langchain.com/oss/python/langgraph/graph-api#runtime-context
    graph = StateGraph(OrchestratorState, context_schema=OrchestratorContext)

    # ---------------------------------------------------------------------------
    # BUILD 2 — Register the four work steps
    # ---------------------------------------------------------------------------
    # add_node(function) uses its function name as the node name. Keep these names stable.
    graph.add_node(triage)
    graph.add_node(stream_guideline_answer)
    graph.add_node(search_catalog)
    graph.add_node(finalize_answer)

    # ---------------------------------------------------------------------------
    # BUILD 3 — Connect both paths
    # ---------------------------------------------------------------------------
    # Clinical-only skips drug lookup. Combined questions stream guidelines before stock.
    # Catalog-only skips guidelines. No route depends on extracted drug classes.
    # START enters the graph; END finishes it.
    # Official reference: https://docs.langchain.com/oss/python/langgraph/graph-api#edges
    graph.add_edge(START, "triage")

    graph.add_conditional_edges(
        "triage",
        lambda state: state["route"],
        {
            "clinical": "stream_guideline_answer",
            "clinical_and_drug": "stream_guideline_answer",
            "drug": "search_catalog",
        },
    )

    graph.add_conditional_edges(
        "stream_guideline_answer",
        lambda state: "search" if state["route"] == "clinical_and_drug" else "no_catalog_list",
        {
            "search": "search_catalog",
            "no_catalog_list": "finalize_answer",
        },
    )

    graph.add_edge("search_catalog", "finalize_answer")

    graph.add_edge("finalize_answer", END)

    # ---------------------------------------------------------------------------
    # BUILD 4 — Compile with the supplied conversation checkpointer
    # ---------------------------------------------------------------------------
    # The caller supplies thread_id when running the compiled graph.
    # Official reference: https://docs.langchain.com/oss/python/langgraph/checkpointers
    return graph.compile(checkpointer=checkpointer)
