"""Connections that every browser tab shares, set up once when the server starts.

Opening a connection to the drug catalog and creating the guidelines store are
slow, and a second copy per tab would waste them (and a second Milvus provider
would disconnect the first). So they are made once here and handed to every tab.
"""

import asyncio
from dataclasses import dataclass
from typing import Literal

from src.adapters.llm_provider import APIType, OpenAIProvider
from src.adapters.vector_store.milvus_vector_store import MilvusProvider
from src.clinical_guideline_agent import ClinicalGuidelineAgent
from src.drugs_agent import DrugAgent, DrugSearcher, verify_drug_collection_ready_for_search
from src.logs import get_component_logger

logger = get_component_logger("HealthcareAssistantWeb")

ConnectionStatus = Literal["connecting", "ready", "failed"]


@dataclass(frozen=True)
class SharedServices:
    """What every tab needs: the drug agent, a chat model, and the guidelines store to use."""

    milvus_provider: MilvusProvider
    chat_completions_provider: OpenAIProvider
    drug_agent: DrugAgent
    guidelines_store_id: str


class SharedServicesManager:
    """Connects the shared services on first use and keeps them for the whole run."""

    def __init__(self) -> None:
        self.status: ConnectionStatus = "connecting"
        self.last_error: str = ""
        self._services: SharedServices | None = None
        self._connect_lock = asyncio.Lock()

    async def get(self) -> SharedServices:
        """Return the shared services, connecting first if that has not happened yet.

        If an earlier attempt failed, the next call tries again.
        """
        async with self._connect_lock:
            if self._services is None:
                self.status = "connecting"
                try:
                    self._services = await self._connect()
                except Exception as error:
                    self.status = "failed"
                    self.last_error = str(error)
                    logger.error("Could not connect shared services", error=str(error))
                    raise
                self.status = "ready"
                logger.info("Shared services ready", store_id=self._services.guidelines_store_id)
            return self._services

    async def warm_up(self) -> None:
        """Connect in the background at startup, so the first visitor does not wait."""
        try:
            await self.get()
        except Exception:
            pass  # already logged; the next page visit will try again

    async def close(self) -> None:
        if self._services is None:
            return
        await self._services.chat_completions_provider.close()
        await self._services.milvus_provider.disconnect()
        self._services = None

    @staticmethod
    async def _connect() -> SharedServices:
        # Step 1: the drug catalog, checked so a missing collection fails here
        # with a clear message rather than on someone's first question.
        milvus_provider = MilvusProvider()
        try:
            await verify_drug_collection_ready_for_search(milvus_provider)
        except Exception:
            await milvus_provider.disconnect()
            raise

        # Step 2: a chat model for the drug agent and the orchestrator.
        chat_completions_provider = OpenAIProvider(api_type=APIType.CHAT_COMPLETION)

        # Step 3: one empty guidelines store for this run. A throwaway agent
        # creates it, which also clears stores left behind by earlier runs.
        setup_agent = ClinicalGuidelineAgent()
        try:
            guidelines_store_id = await setup_agent.start_fresh_guidelines_store()
        except Exception:
            await chat_completions_provider.close()
            await milvus_provider.disconnect()
            raise
        finally:
            await setup_agent.close()

        return SharedServices(
            milvus_provider=milvus_provider,
            chat_completions_provider=chat_completions_provider,
            drug_agent=DrugAgent(DrugSearcher(milvus_provider), chat_completions_provider),
            guidelines_store_id=guidelines_store_id,
        )
