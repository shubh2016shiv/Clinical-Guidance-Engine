# External adapters

This package owns code that communicates with external services. Agent decisions,
tool selection, citation formatting, and medical response policy belong outside it.

| Package | Current implementation | Contract |
| --- | --- | --- |
| `llm_provider` | `openai/` (`provider.py` routes public calls; `chat_completions.py` and `responses_api.py` are the Chat Completions and Responses API clients; `embeddings.py` creates the query embeddings used for vector search). | `base.py` is the existing LLM provider contract; `exceptions.py` holds its errors. |
| `vector_store` | `milvus_vector_store.py`, `openai_vector_store.py` | `base.py` applies to Milvus vector search; `OpenAIVectorStoreClient` handles OpenAI hosted vector stores and files. |
| `redis` | `redis_store.py` | `base.py` holds Redis settings and key construction; `schema.py` holds its config and data models; `exceptions.py` holds its errors. |

The Redis implementation currently supports sessions, response ID chains,
and other cache operations. It is named `redis_store.py` until those concerns have
separate owners; naming the whole implementation `sessions.py` would hide its scope.

Import `OpenAIProvider` from `src.adapters.llm_provider.openai`.
`chat_completions.py` and `responses_api.py` are implementation details of
`OpenAIProvider`.
