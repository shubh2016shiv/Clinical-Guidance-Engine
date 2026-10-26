"""The LangGraph flowchart: which step runs, and in what order.

A "graph" is just a flowchart where each box is a function (a node, defined in
the `nodes` folder) and arrows say which box runs next. This file only wires the
boxes together; the work each box does lives in its own file.

Three paths through the flowchart:

Path A - guideline only
    Ask the guideline agent and stream its answer to the user as it is written.

Path C - drugs only
    Search the catalog for the question itself and send back the drugs found.

Path B - guideline + drugs
    Get the full guideline answer, pull out the drug classes it recommends,
    search the catalog for each class, then write ONE combined answer.
"""

from typing import Any

from langgraph.graph import END, START, StateGraph

from .nodes import (
    make_decide_route_node,
    make_drug_only_answer_node,
    make_extract_drug_classes_node,
    make_fetch_guideline_answer_node,
    make_search_drugs_node,
    make_stream_combined_answer_node,
    make_stream_guideline_answer_node,
)
from .schema import OrchestratorState


def build_orchestrator_graph(
    guideline_agent: Any,
    drug_agent: Any,
    chat_completions_provider: Any,
    model_name: str,
) -> Any:
    """Build and compile the flowchart below. (*) marks a step that streams text.

                                   START
                                     |
                                     v
                            +-----------------+
                            |  decide_route   |   which agents are needed?
                            +-----------------+
                       |             |              |
               "clinical"   "clinical_and_drug"   "drug"
                       |             |              |
                       v             v              v
    +------------------------+  +------------------------+  +------------------+
    | stream_guideline_answer|  | fetch_guideline_answer |  | drug_only_answer |
    |      (*) streams       |  |  whole answer, + [1][2]|  | (*) drug list    |
    +------------------------+  +------------------------+  +------------------+
                       |             |                             |
                       |             v                             |
                       |      +----------------------+             |
                       |      | extract_drug_classes |             |
                       |      +----------------------+             |
                       |             |                             |
                       |             v                             |
                       |      +----------------------+             |
                       |      |     search_drugs     |             |
                       |      |  all classes at once |             |
                       |      +----------------------+             |
                       |             |                             |
                       |             v                             |
                       |      +-----------------------+            |
                       |      | stream_combined_answer|            |
                       |      | (*) streams + list    |            |
                       |      +-----------------------+            |
                       |             |                             |
                       v             v                             v
                                   END
    """
    graph = StateGraph(OrchestratorState)

    # Register every box (node) in the flowchart.
    graph.add_node("decide_route", make_decide_route_node(chat_completions_provider, model_name))
    graph.add_node("stream_guideline_answer", make_stream_guideline_answer_node(guideline_agent))
    graph.add_node("drug_only_answer", make_drug_only_answer_node(drug_agent))
    graph.add_node("fetch_guideline_answer", make_fetch_guideline_answer_node(guideline_agent))
    graph.add_node(
        "extract_drug_classes",
        make_extract_drug_classes_node(chat_completions_provider, model_name),
    )
    graph.add_node("search_drugs", make_search_drugs_node(drug_agent))
    graph.add_node(
        "stream_combined_answer",
        make_stream_combined_answer_node(chat_completions_provider, model_name),
    )

    # Draw the arrows.
    graph.add_edge(START, "decide_route")

    # A three-way fork: the route the model chose names the node to run.
    graph.add_conditional_edges(
        "decide_route",
        lambda state: state["route"],
        {
            "clinical": "stream_guideline_answer",
            "clinical_and_drug": "fetch_guideline_answer",
            "drug": "drug_only_answer",
        },
    )

    # Path A: one step, then done.
    graph.add_edge("stream_guideline_answer", END)

    # Path C: one step, then done.
    graph.add_edge("drug_only_answer", END)

    # Path B: a straight line of steps.
    graph.add_edge("fetch_guideline_answer", "extract_drug_classes")
    graph.add_edge("extract_drug_classes", "search_drugs")
    graph.add_edge("search_drugs", "stream_combined_answer")
    graph.add_edge("stream_combined_answer", END)

    return graph.compile()
