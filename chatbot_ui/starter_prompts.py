"""
Starter Prompts for Chainlit UI.

Provides predefined medical question templates to help users get started
with common healthcare and medication queries.
"""

from typing import List
import chainlit as cl


def get_starter_prompts() -> List[cl.Starter]:
    """
    Get a list of starter prompts for medical questions.

    These prompts provide users with examples of questions they can ask
    and help them understand the capabilities of the healthcare AI assistant.

    Returns:
        List of Chainlit Starter objects with predefined medical questions.
    """
    starters = [
        cl.Starter(
            label="Metformin Formulations",
            message="What are the available formulations and dosages for metformin?",
            icon="/public/pill.svg",
        ),
        cl.Starter(
            label="Warfarin Drug Interactions",
            message="Can you explain the drug interactions for warfarin?",
            icon="/public/warning.svg",
        ),
        cl.Starter(
            label="Type 2 Diabetes Guidelines",
            message="What are the guidelines for treating type 2 diabetes?",
            icon="/public/guidelines.svg",
        ),
        cl.Starter(
            label="ACE Inhibitors Side Effects",
            message="Tell me about the side effects of ACE inhibitors",
            icon="/public/info.svg",
        ),
        cl.Starter(
            label="Hypertension Treatment",
            message="What are the recommended medications for treating hypertension?",
            icon="/public/heart.svg",
        ),
        cl.Starter(
            label="Statin Therapy",
            message="What are the indications and contraindications for statin therapy?",
            icon="/public/medication.svg",
        ),
    ]

    return starters


def get_welcome_starters() -> List[cl.Starter]:
    """
    Get a curated subset of starter prompts for the welcome screen.

    Returns:
        List of 4 most commonly used starter prompts for initial display.
    """
    all_starters = get_starter_prompts()
    # Return first 4 starters for welcome screen
    return all_starters[:4]
