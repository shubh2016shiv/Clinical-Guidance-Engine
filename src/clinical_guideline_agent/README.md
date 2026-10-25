# Clinical guideline agent

This package uses the OpenAI Responses API and its hosted `file_search` tool to
answer questions from clinical guideline files. OpenAI network calls go through
`src.adapters.llm_provider.openai.OpenAIProvider` and
`src.adapters.vector_store.openai_vector_store.OpenAIVectorStoreClient`.
It does not manage the Milvus
drug database or Redis chat sessions. The existing `src/response_api_agent`
package remains available while integrations are migrated.

## Components

| Package | Responsibility |
| --- | --- |
| `responses` | Build requests, parse typed output, handle typed stream events |
| `tools` | Configure hosted `file_search` |
| `vector_store` | Reuse or create a guideline store and wait for file indexing |
| `conversation` | Track the last response ID for one consultation |
| `citations` | Read `file_citation` annotations from output messages |

Use `ClinicalGuidelineAgent.prepare_guidelines_store()` before
`answer_guideline_question()` or `stream_guideline_answer_updates()`. Save the
returned vector store ID if you need to reuse the same store across processes.
Each agent instance keeps one response chain in memory; pass an explicit
`previous_response_id` to resume a saved conversation. Requests explicitly set
`store=True` because response chaining uses stored responses. The agent does
not delete a vector store when closed.

The implementation follows the official OpenAI documentation for
[Responses](https://developers.openai.com/api/docs/guides/migrate-to-responses),
[file search](https://developers.openai.com/api/docs/guides/tools-file-search),
[vector stores](https://developers.openai.com/api/docs/guides/retrieval), and
[streaming events](https://developers.openai.com/api/docs/guides/streaming-responses).
