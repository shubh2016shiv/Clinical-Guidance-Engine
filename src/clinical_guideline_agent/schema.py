"""Data shapes used by the clinical guideline agent."""

from dataclasses import dataclass
from typing import Literal, NotRequired, TypedDict


@dataclass(frozen=True)
class FileCitation:
    file_id: str
    filename: str
    index: int | None = None


@dataclass(frozen=True)
class GuidelineAnswer:
    response_id: str
    text: str
    citations: list[FileCitation]


@dataclass(frozen=True)
class GuidelineStreamUpdate:
    type: Literal["text", "completed"]
    text: str = ""
    response_id: str | None = None
    answer: GuidelineAnswer | None = None


class FileSearchTool(TypedDict):
    type: Literal["file_search"]
    vector_store_ids: list[str]
    max_num_results: NotRequired[int]


class ClinicalGuidelineResponsesApiRequest(TypedDict):
    model: str
    instructions: str
    input: str
    tools: list[FileSearchTool]
    store: bool
    stream: bool
    previous_response_id: NotRequired[str]
