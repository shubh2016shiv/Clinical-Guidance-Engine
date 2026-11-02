# Clinical and drug-stock orchestration

This package chooses capabilities, calls agents, forwards their updates and combines
completed answers. It does not derive pharmacological classes from guideline prose.

## Ownership

| Package | Responsibility |
| --- | --- |
| `src/orchestrator` | Intent routing, node sequence, live updates and answer composition |
| `src/clinical_guideline_agent` | Retrieve guidelines, interpret evidence and supply clinical text/citations |
| `src/drugs_agent` | Identify classes through `drug_classification.py`, retrieve records and stream answers |
| `src/drugs_agent/formatter` | Prepare retrieved records as input tables for the drug agent LLM |
| `src/memory` | Thread configuration, checkpoints, message merging and history selection |
| `src/adapters/llm_provider` | Provider requests and shared structured-reply validation |

## Public imports

```python
from src.orchestrator import ClinicalDrugOrchestrator, OrchestratorAnswer, OrchestratorUpdate
```

The constructor still accepts an injected checkpointer. The stream method emits text
and status updates, followed by one completed answer on success. Exceptions propagate.

## Decide from the question, not from drug classes in an answer

| User intent | Route | What runs |
| --- | --- | --- |
| Treatment, dose, monitoring or safety without a stock request | `clinical` | Guideline agent only |
| Pharmacy stock, forms or strengths only | `drug` | Drug agent with a streamed answer |
| Clinical guidance plus an explicit stock/availability request | `clinical_and_drug` | Guidelines first, then the drug agent streams its answer |

Mentioning a drug in a clinical question does not request stock. Triage resolves
references from history and copies relevant names or queries without expanding class
abbreviations, choosing therapies or interpreting guideline output. If a stock query
has no named product, pass its natural-language stock request rather than inventing one.
The drug classifier may return no class for a condition-only query.

## Graph and numbered nodes

```text
Question -> 01 triage
  clinical          -> 02 guidelines ------------------------> 04 finalize
  clinical_and_drug -> 02 guidelines -> 03 search + stream drugs -> 04 finalize
  drug                              -> 03 search + stream drugs -> 04 finalize
```

These are conditional paths, not parallel graph nodes. Searches within the drug agent
can run concurrently. Filename numbering follows graph-node order. Tutorial workflow
banners use the same node numbers as the filenames. Preparation and completion have
unnumbered banners outside the graph:

| Stage | Read this code |
| --- | --- |
| Prepare question | `orchestrator.py`, using `src/memory` |
| Node 01: choose capabilities and queries | `nodes/node_01_triage.py`, then `triage.py` |
| Node 02: forward clinical guidance | `nodes/node_02_guideline_answer.py` |
| Node 03: forward drug catalog answer | `nodes/node_03_drug_catalog_answer.py` |
| Node 04: cite, compose and remember | `nodes/node_04_final_answer.py`, then `final_answer.py` |
| Deliver completed answer | `orchestrator.py` |

The removed class-planning node is not replaced by another extraction call.
`recommended_drug_classes.py`, `clinical_drug_class_system_prompt.py` and the
`CatalogSearchPlan` schema are removed. No five-class fan-out limit is imposed on the
new query list. Python imposes no five-record display cap.

## Query handoff and compatibility

Node 03 calls `DrugAgent.stream_drug_answer_text(question=..., drug_search_terms=...)`.
Each query goes through the existing drug classifier and search. Searches run concurrently;
results retain input order. Identical rows are combined, while different forms and strengths
remain available. Python tabulates those records before one streamed answer call.
The graph does not call the structured-result API or require a drug class list in state.

The completed answer carries its route, text, sources and optional guideline answer.
Drug text is included in `text`; its section is saved as `catalog_section_text` in graph
state. Empty legacy `drug_classes` and `drug_answer` fields have been removed, and new
conversation log entries no longer write drug classes. Existing log entries are unchanged.
Class identification belongs exclusively to `src/drugs_agent/drug_classification.py`.

## State, streaming and memory

Ordinary state fields are replaced by node updates. The inherited `add_messages` reducer
appends new message IDs and updates matching IDs. `runtime.context` holds agents and
model settings; checkpointed state holds conversation and working results.

Nodes emit custom `kind`/`text` dictionaries through `stream_updates.py`. This uses the
existing single-mode, default v1 streaming shape. The public runner converts events into
`OrchestratorUpdate`; status never enters answer text. Finalization adds guideline citations after streaming. The drug agent's
text chunks are forwarded immediately and kept unchanged in the completed answer.

Memory lives in [src/memory](../memory/README.md). Missing thread IDs create fresh graph
history. Reusing a guideline agent can still reuse its independent Responses API chain;
graph thread IDs do not isolate that chain. Stored agent-result class paths are unchanged.

## Official LangGraph references

- [Graph API: state, reducers, nodes, edges and context](https://docs.langchain.com/oss/python/langgraph/graph-api)
- [Custom streaming and output formats](https://docs.langchain.com/oss/python/langgraph/streaming)
- [Persistence](https://docs.langchain.com/oss/python/langgraph/persistence)
- [Checkpointers, thread IDs and snapshots](https://docs.langchain.com/oss/python/langgraph/checkpointers)

Our routing intent rules and formatting limits are application choices, not LangGraph
requirements. The installed LangGraph 1.2.17 APIs have not been migrated.

## Verification

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_drug_answer_streaming tests.test_retrieved_drug_formatter tests.test_orchestrator tests.test_drugs_agent tests.test_clinical_guideline_agent tests.test_memory tests.test_inline_citations tests.test_app tests.test_web_ui_services
.\.venv\Scripts\python.exe -m ruff check src/orchestrator
.\.venv\Scripts\python.exe -m ruff format --check src/orchestrator
```

Tests exercise mocked routing decisions and both real and mocked agent boundaries.
They verify clinical-only/stock-only/combined paths, classifier invocation, unexpanded
queries, ordered parallel searches, failures, history, citations and immediate streaming.
They do not measure live-model routing accuracy. App imports require access to the
existing log directory outside the workspace.

## LLM input table and streamed output

The table is context for the drug agent's LLM, not the final answer. The agent selects
relevant retrieved rows and streams concise prose or bullets. Empty retrieval also goes
to that call so the agent can answer the question consistently. The orchestrator adds a
separator once on the combined route, forwards chunks, and stores their joined text.

No extra stock-rendering node or post-generation formatting call is needed. The old
Python stock-table formatter is removed. The streamed answer is grounded through prompt
instructions; it is not checked by a structured drug-name validator. Tests verify the request and stream mechanics, not live-model factual accuracy.
