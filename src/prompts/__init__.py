"""
System prompts for the healthcare AI agent.
"""

from src.prompts.clinical_guideline_system_prompt import (
    CLINICAL_GUIDELINE_SYSTEM_PROMPT,
)
from src.prompts.drug_classification_system_prompt import (
    DRUG_CLASSIFICATION_SYSTEM_PROMPT,
)
from src.prompts.drug_system_prompt import DRUG_STREAM_SYSTEM_PROMPT
from src.prompts.orchestrator_system_prompts import (
    QUESTION_TRIAGE_SYSTEM_PROMPT,
)

__all__ = [
    "CLINICAL_GUIDELINE_SYSTEM_PROMPT",
    "DRUG_CLASSIFICATION_SYSTEM_PROMPT",
    "DRUG_STREAM_SYSTEM_PROMPT",
    "QUESTION_TRIAGE_SYSTEM_PROMPT",
]
