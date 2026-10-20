"""Example questions supported by the available clinical guidelines."""

import chainlit as cl


def get_welcome_starters() -> list[cl.Starter]:
    return [
        cl.Starter(
            label="Hypertension guidelines",
            message="What lifestyle recommendations are in the hypertension guideline?",
        ),
        cl.Starter(
            label="Diabetes guidelines",
            message="What topics are covered by the diabetes guideline?",
        ),
        cl.Starter(
            label="Drug interactions",
            message="What does the drug interactions guideline say about checking interactions?",
        ),
    ]
