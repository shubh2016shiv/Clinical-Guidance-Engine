"""Everything that belongs to one browser tab: its agents and its conversation."""

from uuid import uuid4

from src.clinical_guideline_agent import ClinicalGuidelineAgent
from src.orchestrator import ClinicalDrugOrchestrator

from .answer_streamer import AnswerStreamer
from .conversation_log import ConversationLog
from .guideline_library import GuidelineLibrary
from .shared_services import SharedServices, SharedServicesManager


class UserSession:
    """One tab's conversation. Tabs share the guidelines store but not the conversation."""

    def __init__(self, shared_services: SharedServices) -> None:
        self.session_id = str(uuid4())
        # Each tab gets its own agent so follow-up questions only see that tab's
        # earlier answers, but every agent points at the same guidelines store.
        self.guideline_agent = ClinicalGuidelineAgent(
            guidelines_store_id=shared_services.guidelines_store_id
        )
        orchestrator = ClinicalDrugOrchestrator(
            self.guideline_agent,
            shared_services.drug_agent,
            shared_services.chat_completions_provider,
        )
        self.guideline_library = GuidelineLibrary(self.guideline_agent)
        self.answer_streamer = AnswerStreamer(orchestrator, self.session_id, ConversationLog())

    async def close(self) -> None:
        await self.guideline_agent.close()


class LazyUserSession:
    """Creates a tab's UserSession the first time something needs it.

    The page can be drawn straight away; this waits for the shared connections
    only when someone uploads a file or asks a question.
    """

    def __init__(self, shared_services_manager: SharedServicesManager) -> None:
        self.shared_services_manager = shared_services_manager
        self._user_session: UserSession | None = None

    async def get(self) -> UserSession:
        if self._user_session is None:
            shared_services = await self.shared_services_manager.get()
            # Check again: another click may have created it while we were waiting.
            if self._user_session is None:
                self._user_session = UserSession(shared_services)
        return self._user_session

    async def restart(self) -> None:
        """Throw away this tab's conversation so the next question starts fresh.

        The guideline agent remembers earlier answers to follow up on them, so a
        new conversation needs a new agent. The shared guidelines store is kept.
        """
        await self.close()

    async def close(self) -> None:
        if self._user_session is not None:
            await self._user_session.close()
            self._user_session = None
