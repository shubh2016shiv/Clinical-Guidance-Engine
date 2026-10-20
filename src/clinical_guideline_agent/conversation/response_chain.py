"""Track the Responses API conversation ID for one guideline consultation."""


class ResponsesApiConversationState:
    """Keep the latest response ID in memory for the next Responses API call."""

    def __init__(self) -> None:
        self.latest_responses_api_response_id: str | None = None

    def record_completed_response_id(self, response_id: str) -> None:
        if not response_id:
            raise ValueError("A completed response must have an ID")
        self.latest_responses_api_response_id = response_id
