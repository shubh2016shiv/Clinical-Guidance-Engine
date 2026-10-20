"""Chainlit interface for clinical guideline questions."""

import sys
from pathlib import Path
from uuid import uuid4

# Chainlit loads this file by path, so make the project package importable.
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import chainlit as cl  # noqa: E402

from healthcare_assistant_ui.chat_handler import ChainlitChatHandler  # noqa: E402
from healthcare_assistant_ui.citation_formatter import ChainlitCitationFormatter  # noqa: E402
from healthcare_assistant_ui.config import (  # noqa: E402
    ERROR_MESSAGE_AGENT_NOT_READY,
    ERROR_MESSAGE_INITIALIZATION,
    ERROR_MESSAGE_PROCESSING,
    SHOW_CITATION_ELEMENTS,
    STREAM_RESPONSES,
    WELCOME_MESSAGE,
)
from healthcare_assistant_ui.persistence_manager import (  # noqa: E402
    ConversationPersistenceManager,
)
from healthcare_assistant_ui.starter_prompts import get_welcome_starters  # noqa: E402
from src.clinical_guideline_agent import ClinicalGuidelineAgent  # noqa: E402
from src.logs import get_component_logger  # noqa: E402

logger = get_component_logger("HealthcareAssistantUI")
chat_handler = ChainlitChatHandler(
    persistence_manager=ConversationPersistenceManager(),
    citation_formatter=ChainlitCitationFormatter(),
)


@cl.set_starters
async def set_starters():
    return get_welcome_starters()


@cl.on_chat_start
async def on_chat_start() -> None:
    initialization_message = cl.Message(content="Preparing clinical guidelines...")
    await initialization_message.send()

    guideline_agent = None
    try:
        guideline_agent = ClinicalGuidelineAgent()
        guidelines_store_id = await guideline_agent.prepare_guidelines_store()
        session_id = str(uuid4())

        cl.user_session.set("session_id", session_id)
        cl.user_session.set("guideline_agent", guideline_agent)
        cl.user_session.set("guidelines_store_id", guidelines_store_id)

        initialization_message.content = "Clinical guidelines are ready."
        await initialization_message.update()
        await cl.Message(content=WELCOME_MESSAGE).send()
        logger.info("Guideline chat ready", session_id=session_id)
    except Exception as error:
        if guideline_agent is not None:
            await guideline_agent.close()
        logger.error("Failed to prepare guideline chat", error=str(error))
        initialization_message.content = ERROR_MESSAGE_INITIALIZATION
        await initialization_message.update()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    guideline_agent = cl.user_session.get("guideline_agent")
    session_id = cl.user_session.get("session_id")
    if guideline_agent is None or session_id is None:
        await cl.Message(content=ERROR_MESSAGE_AGENT_NOT_READY).send()
        return

    response_message = cl.Message(content="")
    await response_message.send()

    try:
        if STREAM_RESPONSES:
            citations = []
            async for update in chat_handler.handle_message(
                message=message.content,
                agent=guideline_agent,
                session_id=session_id,
            ):
                if update.get("chunk"):
                    await response_message.stream_token(update["chunk"])
                if update.get("is_complete"):
                    citations = update.get("citations", [])
        else:
            answer = await chat_handler.handle_message_non_streaming(
                message=message.content,
                agent=guideline_agent,
                session_id=session_id,
            )
            response_message.content = answer["content"]
            citations = answer["citations"]

        if citations and SHOW_CITATION_ELEMENTS:
            response_message.content += chat_handler.citation_formatter.format_citations_section(
                citations
            )
            response_message.elements = chat_handler.citation_formatter.create_citation_elements(
                citations
            )
        await response_message.update()
    except Exception as error:
        logger.error("Guideline answer failed", session_id=session_id, error=str(error))
        response_message.content = ERROR_MESSAGE_PROCESSING
        await response_message.update()


@cl.on_chat_end
async def on_chat_end() -> None:
    guideline_agent = cl.user_session.get("guideline_agent")
    if guideline_agent is not None:
        await guideline_agent.close()
        cl.user_session.set("guideline_agent", None)
