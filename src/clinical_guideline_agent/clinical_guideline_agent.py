"""Answer questions using clinical guideline files and Responses file search."""

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from src.adapters.llm_provider import APIType, OpenAIProvider
from src.adapters.vector_store.openai_vector_store import OpenAIVectorStoreClient
from src.config import PROJECT_ROOT, get_settings
from src.prompts.clinical_guideline_system_prompt import CLINICAL_GUIDELINE_SYSTEM_PROMPT

from .conversation.response_chain import ResponsesApiConversationState
from .responses.create_response import ClinicalGuidelineResponsesApi
from .responses.parse_response import parse_responses_api_result_into_guideline_answer
from .responses.stream_response import stream_guideline_updates_from_responses_api_events
from .schema import GuidelineAnswer, GuidelineStreamUpdate, IndexedGuidelineFile
from .vector_store.guidelines_store import GuidelinesStore


class ClinicalGuidelineAgent:
    """Answer guideline questions with one Responses API conversation per instance."""

    def __init__(
        self,
        *,
        responses_api_provider: Any | None = None,
        openai_vector_store_client: Any | None = None,
        model_name: str | None = None,
        guideline_instructions: str = CLINICAL_GUIDELINE_SYSTEM_PROMPT,
        guidelines_store_id: str | None = None,
    ) -> None:
        application_settings = get_settings()
        if (
            responses_api_provider is None or openai_vector_store_client is None
        ) and not application_settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required")
        self._owns_responses_api_provider = responses_api_provider is None
        self._owns_openai_vector_store_client = openai_vector_store_client is None
        self.responses_api_provider = (
            responses_api_provider
            if responses_api_provider is not None
            else OpenAIProvider(
                api_type=APIType.RESPONSES,
                model_name=model_name or application_settings.openai_model_name,
            )
        )
        self.openai_vector_store_client = (
            openai_vector_store_client
            if openai_vector_store_client is not None
            else OpenAIVectorStoreClient()
        )
        self.responses_api = ClinicalGuidelineResponsesApi(
            self.responses_api_provider,
            model=model_name or application_settings.openai_model_name,
            instructions=guideline_instructions,
        )
        self.guidelines_store = GuidelinesStore(self.openai_vector_store_client)
        self.guidelines_store_id = guidelines_store_id
        self.responses_api_conversation_state = ResponsesApiConversationState()

    async def prepare_guidelines_store(
        self,
        guidelines_directory: str | Path | None = None,
        *,
        guidelines_store_id: str | None = None,
    ) -> str:
        """Find or prepare the hosted store containing clinical guidelines."""
        application_settings = get_settings()
        guidelines_directory_path = (
            guidelines_directory
            or PROJECT_ROOT / application_settings.clinical_guidelines_directory
        )
        self.guidelines_store_id = await self.guidelines_store.ensure_ready(
            guidelines_directory_path,
            vector_store_id=guidelines_store_id or self.guidelines_store_id,
        )
        return self.guidelines_store_id

    async def start_fresh_guidelines_store(self) -> str:
        """Discard the previous store and start with an empty one for this run.

        Used by the chat UI, where guidelines arrive as chat attachments rather
        than from a local directory, so nothing is uploaded until a user does it.
        """
        self.guidelines_store_id = await self.guidelines_store.start_fresh(self.guidelines_store_id)
        return self.guidelines_store_id

    async def add_guideline_files(
        self,
        guideline_file_paths: list[Path],
        *,
        display_names: dict[Path, str] | None = None,
    ) -> Any:
        """Index guideline files a user attached in the chat.

        ``display_names`` gives the original file name for each path, so citations
        show it rather than the name the chat UI stored the upload under.
        """
        return await self.guidelines_store.add_files(
            self._require_guidelines_store_id(),
            guideline_file_paths,
            display_names=display_names,
        )

    async def list_guideline_files(self) -> list[IndexedGuidelineFile]:
        """List the guideline files currently indexed in this agent's store."""
        return await self.guidelines_store.list_files(self._require_guidelines_store_id())

    async def remove_guideline_file(self, file_id: str) -> None:
        """Remove one guideline file from this agent's store and delete the upload."""
        await self.guidelines_store.remove_file(self._require_guidelines_store_id(), file_id)

    async def answer_guideline_question(
        self,
        guideline_question: str,
        *,
        previous_response_id: str | None = None,
        max_file_search_results: int | None = None,
    ) -> GuidelineAnswer:
        """Return a guideline answer with its response ID and file citations."""
        responses_api_result = await self.responses_api.create_responses_api_result(
            guideline_question,
            vector_store_id=self._require_guidelines_store_id(),
            previous_response_id=(
                previous_response_id
                or self.responses_api_conversation_state.latest_responses_api_response_id
            ),
            max_results=max_file_search_results,
        )
        guideline_answer = parse_responses_api_result_into_guideline_answer(responses_api_result)
        self.responses_api_conversation_state.record_completed_response_id(
            guideline_answer.response_id
        )
        return guideline_answer

    async def stream_guideline_answer_updates(
        self,
        guideline_question: str,
        *,
        previous_response_id: str | None = None,
        max_file_search_results: int | None = None,
    ) -> AsyncIterator[GuidelineStreamUpdate]:
        """Yield text updates followed by the completed guideline answer."""
        responses_api_events = await self.responses_api.start_responses_api_stream(
            guideline_question,
            vector_store_id=self._require_guidelines_store_id(),
            previous_response_id=(
                previous_response_id
                or self.responses_api_conversation_state.latest_responses_api_response_id
            ),
            max_results=max_file_search_results,
        )
        async for guideline_update in stream_guideline_updates_from_responses_api_events(
            responses_api_events
        ):
            if guideline_update.answer is not None:
                self.responses_api_conversation_state.record_completed_response_id(
                    guideline_update.answer.response_id
                )
            yield guideline_update

    async def close(self) -> None:
        """Close clients created by this agent without deleting the store."""
        if self._owns_responses_api_provider:
            await self.responses_api_provider.close()
        if self._owns_openai_vector_store_client:
            await self.openai_vector_store_client.close()

    def _require_guidelines_store_id(self) -> str:
        if not self.guidelines_store_id:
            raise RuntimeError("Call prepare_guidelines_store() before asking guideline questions")
        return self.guidelines_store_id
