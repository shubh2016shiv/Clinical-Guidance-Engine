# Bootstrap

Startup checklist for the healthcare assistant. Connects to each optional service (Redis, Milvus, clinical guidelines), degrades gracefully when one is unavailable, and hands the rest of the app a single result describing what's actually usable this run.

## What This Module Does

Before the app can answer a question, it needs some combination of:
- **Redis** — session/chat caching
- **Milvus** — the drug vector database, for drug search
- **OpenAI file search** — the clinical guidelines store, for citations

Any of these can be down, unconfigured, or not yet populated. `bootstrap_services()` tries each one, and instead of crashing the whole app when one fails, it records a warning and leaves that feature disabled — so a user can still get a drug-search answer even if the guidelines store is unreachable, or still get guideline citations even if Redis is down.

**Who uses this:** [src/app.py](../app.py) calls `bootstrap_services()` at the start of every CLI invocation, before routing to the drug agent, guideline agent, or orchestrator.

## Architecture

```
bootstrap_services()
        │
        ├─ check_settings()           Step 1: fail fast on missing config
        │                             (no network calls — pure validation)
        │
        ├─ check_redis()              Step 2: always attempted
        │    └─ RedisProvider.connect()
        │
        ├─ check_milvus()             Step 3: only if use_drug_database=True
        │    └─ verify_drug_collection_ready_for_search()
        │
        └─ prepare_guidelines_store() Step 4: only if use_clinical_guidelines=True
             └─ ClinicalGuidelineAgent.prepare_guidelines_store()

        → BootstrapResult(redis_cache, drug_provider, guideline_agent, warnings)
```

Each step is independent: one failing does not stop the others from being attempted.

## Quick Start

### Basic usage

```python
from src.bootstrap import bootstrap_services

startup = await bootstrap_services()
try:
    if startup.drug_provider is None:
        print("Drug search unavailable:", startup.warnings)
    else:
        # use startup.drug_provider, startup.redis_cache, startup.guideline_agent
        ...
finally:
    await startup.close()
```

### Turning off a feature you don't need

Skip a service's checks entirely if a given run doesn't use it — this avoids requiring Milvus or guideline files to be configured just to answer a drug-only or guideline-only question:

```python
# Drug-only question: don't bother checking clinical guidelines.
startup = await bootstrap_services(use_drug_database=True, use_clinical_guidelines=False)

# Guideline-only question: don't bother checking Milvus.
startup = await bootstrap_services(use_drug_database=False, use_clinical_guidelines=True)
```

This is how [src/app.py:18-21](../app.py#L18-L21) picks which checks to run based on `--source`.

### Checking what's available

```python
startup = await bootstrap_services()

if startup.guideline_agent is None:
    raise RuntimeError(f"Clinical guidelines are unavailable. Warnings: {startup.warnings}")

answer = await startup.guideline_agent.answer_guideline_question("What is metformin?")
```

### Cleaning up

Always close the result when done, even on failure — this releases Redis and Milvus connections and closes the guideline agent's HTTP client:

```python
startup = await bootstrap_services()
try:
    ...
finally:
    await startup.close()
```

`close()` is safe to call even if some services never started — it only closes what's not `None`.

## BootstrapResult

| Field | Type | Meaning |
| --- | --- | --- |
| `redis_cache` | `RedisProvider \| None` | Connected Redis provider, or `None` if Redis was unreachable |
| `drug_provider` | `MilvusProvider \| None` | Connected Milvus client, or `None` if drug search isn't ready |
| `guideline_agent` | `ClinicalGuidelineAgent \| None` | Ready guideline agent, or `None` if the guidelines store couldn't be prepared |
| `warnings` | `list[str]` | Human-readable reasons for anything that was skipped |

A `None` field means that feature is disabled for this run — always check before using `drug_provider`, `redis_cache`, or `guideline_agent`.

## Failure Behavior

| Check | On Failure | App Behavior |
| --- | --- | --- |
| `check_settings()` | Raises `ValueError` immediately | App startup aborts — this is a config bug, not a transient outage |
| `check_redis()` | Returns `False`, logs a warning | `redis_cache` is `None`; session caching disabled, everything else continues |
| `check_milvus()` | Returns `False`, logs a warning, disconnects | `drug_provider` is `None`; drug search disabled |
| `prepare_guidelines_store()` | Returns `None`, logs a warning | `guideline_agent` is closed and set to `None`; guideline citations disabled |

**Why settings validation is different:** a missing `OPENAI_API_KEY` or misconfigured `REDIS_HOST` means the deployment itself is broken — there's nothing to gracefully degrade to, so `check_settings()` raises instead of warning. Redis/Milvus/guidelines failures, by contrast, are treated as *this instance of this service* being temporarily unavailable, which the app can route around.

## Required Settings

`check_settings()` validates these before anything connects:

| Setting | Always Required | Required When |
| --- | --- | --- |
| `OPENAI_API_KEY` | ✅ | — |
| `OPENAI_MODEL_NAME` | ✅ | — |
| `REDIS_HOST` | ✅ | — |
| `MILVUS_HOST`, `MILVUS_COLLECTION_NAME`, `OPENAI_EMBEDDING_MODEL` | — | `use_drug_database=True` |
| `CLINICAL_GUIDELINES_DIRECTORY` (must exist as a directory) | — | `use_clinical_guidelines=True` |

## File Reference

| File | Purpose |
| --- | --- |
| [__init__.py](__init__.py) | `bootstrap_services()`, `BootstrapResult` — the entry point and result container |
| [check_settings.py](check_settings.py) | Validates required config exists before any connection is attempted |
| [check_redis_and_milvus.py](check_redis_and_milvus.py) | `check_redis()`, `check_milvus()` — connection + readiness checks |
| [prepare_guidelines_store.py](prepare_guidelines_store.py) | Reuses or creates the OpenAI file-search store for clinical guidelines |

## Adding a New Service

To add another optional dependency (e.g., a second vector store):

1. Write a `check_<service>()` function following the pattern in `check_redis_and_milvus.py`: try to connect/verify, catch the service's specific exception type, log a warning, return `False`.
2. Add any required settings to `check_settings.py`, gated behind a feature flag if the service is optional.
3. Add a field to `BootstrapResult` and close it in `BootstrapResult.close()`.
4. Wire it into `bootstrap_services()` following the same `if await check_<service>(...): result.<field> = ... else: result.warnings.append(...)` shape.

## Glossary

- **Bootstrap:** the startup sequence that prepares external services before the app can serve requests.
- **Graceful degradation:** continuing to run with reduced functionality instead of crashing when a non-critical dependency fails.
- **Feature flag:** a boolean (`use_drug_database`, `use_clinical_guidelines`) that skips a service's checks entirely when that feature isn't needed for the current run.
