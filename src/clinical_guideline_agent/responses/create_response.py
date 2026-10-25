"""Send clinical guideline requests to the Responses API."""

from typing import Any

from ..schema import ClinicalGuidelineResponsesApiRequest
from ..tools.file_search import file_search_tool


class ClinicalGuidelineResponsesApi:
    """Build and send clinical guideline requests through the Responses API."""

    def __init__(self, responses_api_provider: Any, *, model: str, instructions: str) -> None:
        self.responses_api_provider = responses_api_provider
        self.model = model
        self.instructions = instructions

    def build_responses_api_request(
        self,
        question: str,
        *,
        vector_store_id: str,
        previous_response_id: str | None = None,
        max_results: int | None = None,
        stream: bool = False,
    ) -> ClinicalGuidelineResponsesApiRequest:
        """Build a Responses request with guidance repeated on every turn."""
        if not question.strip():
            raise ValueError("A question is required")
        responses_api_request: ClinicalGuidelineResponsesApiRequest = {
            "model": self.model,
            "instructions": self.instructions,
            "input": question,
            "tools": [file_search_tool(vector_store_id, max_results=max_results)],
            "store": True,
            "stream": stream,
        }
        if previous_response_id:
            responses_api_request["previous_response_id"] = previous_response_id
        return responses_api_request

    async def create_responses_api_result(self, question: str, **kwargs: Any) -> Any:
        responses_api_request = self.build_responses_api_request(question, **kwargs)
        return await self.responses_api_provider.execute_responses_api(
            input_message=responses_api_request.pop("input"), **responses_api_request
        )

    async def start_responses_api_stream(self, question: str, **kwargs: Any) -> Any:
        responses_api_request = self.build_responses_api_request(question, stream=True, **kwargs)
        return await self.responses_api_provider.execute_responses_api(
            input_message=responses_api_request.pop("input"), **responses_api_request
        )
