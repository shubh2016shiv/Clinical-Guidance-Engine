"""
System prompts for the healthcare AI agent.
"""

from src.prompts.asclepius_system_prompt import (
    HEALTHCARE_AGENT_SYSTEM_PROMPT,
    get_system_prompt,
)
from src.prompts.clinical_guideline_system_prompt import (
    CLINICAL_GUIDELINE_SYSTEM_PROMPT,
)

__all__ = [
    "get_system_prompt",
    "HEALTHCARE_AGENT_SYSTEM_PROMPT",
    "CLINICAL_GUIDELINE_SYSTEM_PROMPT",
]
