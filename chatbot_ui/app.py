"""
Main Chainlit Application for Drug Recommendation Chatbot.

Provides a professional web-based UI for the Asclepius Healthcare AI Assistant,
featuring real-time streaming, citation display, and conversation persistence.
"""

import sys
from pathlib import Path

# Add project root to Python path to enable src imports
project_root = Path(__file__).parent.parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import uuid  # noqa: E402
import chainlit as cl  # noqa: E402
from src.response_api_agent.asclepius_agent import AsclepiusHealthcareAgent  # noqa: E402
from src.config import get_settings  # noqa: E402
from src.logs import get_component_logger  # noqa: E402
from chatbot_ui.chat_handler import ChainlitChatHandler  # noqa: E402
from chatbot_ui.persistence_manager import ConversationPersistenceManager  # noqa: E402
from chatbot_ui.citation_formatter import ChainlitCitationFormatter  # noqa: E402
from chatbot_ui.starter_prompts import get_welcome_starters  # noqa: E402
from chatbot_ui.config import (  # noqa: E402
    WELCOME_MESSAGE,
    ERROR_MESSAGE_INITIALIZATION,
    ERROR_MESSAGE_PROCESSING,
    ERROR_MESSAGE_AGENT_NOT_READY,
    AGENT_CHAT_HISTORY_LIMIT,
    ENABLE_CLINICAL_GUIDELINES,
    ENABLE_DRUG_DATABASE,
    STREAM_RESPONSES,
    SHOW_CITATION_ELEMENTS,
)


# Initialize global components
logger = get_component_logger("ChainlitUI")
settings = get_settings()
chat_handler = ChainlitChatHandler(
    persistence_manager=ConversationPersistenceManager(),
    citation_formatter=ChainlitCitationFormatter(),
)


@cl.set_starters
async def set_starters():
    """
    Set starter prompts for the chat interface.

    Provides users with example questions to help them get started.
    """
    return get_welcome_starters()


@cl.on_chat_start
async def on_chat_start():
    """
    Initialize the chat session when a user connects.

    Sets up:
    - Session ID with smart resumption of existing sessions
    - AsclepiusHealthcareAgent with knowledge base
    - Welcome message with starter prompts

    New in optimized version:
    - Attempts to resume the latest active session if available
    - Creates new session only if no active session exists
    - Reduces unnecessary vector store initialization
    """
    # Display initialization message
    init_msg = cl.Message(content="Initializing Asclepius Healthcare AI Assistant...")
    await init_msg.send()

    session_id = None
    try:
        # Initialize the healthcare agent
        agent = AsclepiusHealthcareAgent(chat_history_limit=AGENT_CHAT_HISTORY_LIMIT)

        # Setup knowledge base with clinical guidelines
        logger.info(
            "Setting up clinical knowledge base",
            component="ChainlitUI",
            subcomponent="OnChatStart",
        )

        vector_store_id = await agent.initialize_knowledge_base()

        # Get or create session using smart resumption
        # This will try to resume the latest active session or create a new one
        # Access chat_manager through response_manager -> OpenAIResponseManager.chat_manager
        try:
            if (
                agent.response_manager
                and hasattr(agent.response_manager, "chat_manager")
                and agent.response_manager.chat_manager
            ):
                session_id = (
                    await agent.response_manager.chat_manager.get_or_create_session(
                        session_id=None,  # Let it auto-detect
                        vector_store_id=vector_store_id,
                        resume_latest=True,  # Enable smart session resumption
                    )
                )
                logger.info(
                    "Session acquired through smart resumption",
                    component="ChainlitUI",
                    subcomponent="OnChatStart",
                    session_id=session_id,
                )
            else:
                # Fallback if response_manager or chat_manager not available
                session_id = str(uuid.uuid4())
                logger.warning(
                    "Response manager or chat_manager not available, using UUID session",
                    component="ChainlitUI",
                    subcomponent="OnChatStart",
                    session_id=session_id,
                )
        except Exception as e:
            # If session resumption fails, fall back to UUID
            session_id = str(uuid.uuid4())
            logger.warning(
                f"Session resumption failed, falling back to UUID: {e}",
                component="ChainlitUI",
                subcomponent="OnChatStart",
                session_id=session_id,
                error=str(e),
            )

        cl.user_session.set("session_id", session_id)

        logger.info(
            "Chat session initialized",
            component="ChainlitUI",
            subcomponent="OnChatStart",
            session_id=session_id,
        )

        # Store agent in session
        cl.user_session.set("agent", agent)
        cl.user_session.set("agent_ready", True)
        cl.user_session.set("vector_store_id", vector_store_id)

        if vector_store_id:
            logger.info(
                "Knowledge base initialized successfully",
                component="ChainlitUI",
                subcomponent="OnChatStart",
                session_id=session_id,
                vector_store_id=vector_store_id,
            )

            # Update initialization message
            init_msg.content = "Knowledge base loaded successfully!"
            await init_msg.update()
        else:
            logger.warning(
                "Knowledge base initialization incomplete",
                component="ChainlitUI",
                subcomponent="OnChatStart",
                session_id=session_id,
            )

            # Update initialization message
            init_msg.content = "Initialized without clinical guidelines database."
            await init_msg.update()

        # Send welcome message
        welcome_msg = cl.Message(content=WELCOME_MESSAGE)
        await welcome_msg.send()

        logger.info(
            "Chat session ready",
            component="ChainlitUI",
            subcomponent="OnChatStart",
            session_id=session_id,
        )

    except Exception as e:
        # Fallback: create UUID session if smart resumption fails
        if not session_id:
            session_id = str(uuid.uuid4())
        cl.user_session.set("session_id", session_id)

        logger.error(
            "Failed to initialize chat session, using fallback UUID session",
            component="ChainlitUI",
            subcomponent="OnChatStart",
            session_id=session_id,
            error=str(e),
            error_type=type(e).__name__,
        )

        # Update initialization message with error
        init_msg.content = ERROR_MESSAGE_INITIALIZATION
        await init_msg.update()

        # Send error message
        error_msg = cl.Message(
            content=f"Error details: {str(e)}\n\nPlease refresh the page to try again."
        )
        await error_msg.send()

        # Mark agent as not ready
        cl.user_session.set("agent_ready", False)


@cl.on_message
async def on_message(message: cl.Message):
    """
    Handle incoming user messages.

    Processes the message through the agent and streams the response back to the user,
    including citations and metadata.

    Args:
        message: The user's message from Chainlit.
    """
    session_id = cl.user_session.get("session_id")
    agent = cl.user_session.get("agent")
    agent_ready = cl.user_session.get("agent_ready", False)

    logger.info(
        "Received user message",
        component="ChainlitUI",
        subcomponent="OnMessage",
        session_id=session_id,
        message_length=len(message.content),
    )

    # Check if agent is ready
    if not agent_ready or agent is None:
        logger.warning(
            "Agent not ready for message processing",
            component="ChainlitUI",
            subcomponent="OnMessage",
            session_id=session_id,
        )

        error_msg = cl.Message(content=ERROR_MESSAGE_AGENT_NOT_READY)
        await error_msg.send()
        return

    # Create response message for streaming
    response_msg = cl.Message(content="")

    try:
        if STREAM_RESPONSES:
            # Stream the response
            citations = []
            response_id = None

            async for chunk_data in chat_handler.handle_message(
                message=message.content,
                agent=agent,
                session_id=session_id,
                use_clinical_guidelines=ENABLE_CLINICAL_GUIDELINES,
                use_drug_database=ENABLE_DRUG_DATABASE,
            ):
                chunk_text = chunk_data.get("chunk", "")
                response_id = chunk_data.get("response_id") or response_id
                is_citation = chunk_data.get("is_citation", False)
                is_complete = chunk_data.get("is_complete", False)

                # Collect citations from citation chunks or final completion chunk
                chunk_citations = chunk_data.get("citations", [])
                if chunk_citations:
                    citations.extend(chunk_citations)

                # Stream text chunks (excluding citations)
                if chunk_text and not is_citation:
                    await response_msg.stream_token(chunk_text)

                # Check for errors in completion chunk
                if is_complete and chunk_data.get("error"):
                    logger.error(
                        "Error during message processing",
                        component="ChainlitUI",
                        subcomponent="OnMessage",
                        session_id=session_id,
                        error=chunk_data.get("error"),
                    )

            # Add citations as elements if available
            if citations and SHOW_CITATION_ELEMENTS:
                citation_elements = (
                    chat_handler.citation_formatter.create_citation_elements(citations)
                )

                if citation_elements:
                    # Append citations section to the message
                    citations_section = (
                        chat_handler.citation_formatter.format_citations_section(
                            citations
                        )
                    )
                    response_msg.content += citations_section

                    # Add citation elements to message
                    response_msg.elements = citation_elements

                    logger.info(
                        "Citations added to response",
                        component="ChainlitUI",
                        subcomponent="OnMessage",
                        session_id=session_id,
                        citation_count=len(citations),
                    )

            # Send the complete message
            await response_msg.send()

            logger.info(
                "Message processing completed",
                component="ChainlitUI",
                subcomponent="OnMessage",
                session_id=session_id,
                response_id=response_id,
                citations_count=len(citations),
            )

        else:
            # Non-streaming fallback
            result = await chat_handler.handle_message_non_streaming(
                message=message.content,
                agent=agent,
                session_id=session_id,
                use_clinical_guidelines=ENABLE_CLINICAL_GUIDELINES,
                use_drug_database=ENABLE_DRUG_DATABASE,
            )

            response_content = result.get("content", "")
            citations = result.get("citations", [])
            response_id = result.get("response_id")

            # Format response with citations
            if citations and SHOW_CITATION_ELEMENTS:
                formatted_content, citation_elements = (
                    chat_handler.format_response_with_citations(
                        response_content, citations
                    )
                )
                response_msg.content = formatted_content
                response_msg.elements = citation_elements
            else:
                response_msg.content = response_content

            await response_msg.send()

            logger.info(
                "Message processing completed (non-streaming)",
                component="ChainlitUI",
                subcomponent="OnMessage",
                session_id=session_id,
                response_id=response_id,
                citations_count=len(citations),
            )

    except Exception as e:
        logger.error(
            "Unexpected error during message processing",
            component="ChainlitUI",
            subcomponent="OnMessage",
            session_id=session_id,
            error=str(e),
            error_type=type(e).__name__,
        )

        # Send error message
        response_msg.content = ERROR_MESSAGE_PROCESSING
        await response_msg.send()

        # Send error details
        error_details_msg = cl.Message(content=f"Technical details: {str(e)}")
        await error_details_msg.send()


@cl.on_chat_end
async def on_chat_end():
    """
    Handle chat session cleanup when user disconnects.

    Performs graceful shutdown of the agent and releases resources.
    """
    session_id = cl.user_session.get("session_id")
    agent = cl.user_session.get("agent")

    logger.info(
        "Chat session ending",
        component="ChainlitUI",
        subcomponent="OnChatEnd",
        session_id=session_id,
    )

    if agent:
        try:
            # Close agent session based on settings
            if settings.enable_cleanup:
                logger.info(
                    "Cleaning up agent resources",
                    component="ChainlitUI",
                    subcomponent="OnChatEnd",
                    session_id=session_id,
                )

                await agent.close_session()

                logger.info(
                    "Agent resources cleaned up successfully",
                    component="ChainlitUI",
                    subcomponent="OnChatEnd",
                    session_id=session_id,
                )
            else:
                logger.info(
                    "Cleanup disabled, resources will persist",
                    component="ChainlitUI",
                    subcomponent="OnChatEnd",
                    session_id=session_id,
                )
        except Exception as e:
            logger.error(
                "Error during session cleanup",
                component="ChainlitUI",
                subcomponent="OnChatEnd",
                session_id=session_id,
                error=str(e),
                error_type=type(e).__name__,
            )

    logger.info(
        "Chat session ended",
        component="ChainlitUI",
        subcomponent="OnChatEnd",
        session_id=session_id,
    )


@cl.on_stop
async def on_stop():
    """
    Handle user stopping the current response generation.

    Called when user clicks stop button during streaming.
    """
    session_id = cl.user_session.get("session_id")

    logger.info(
        "User stopped response generation",
        component="ChainlitUI",
        subcomponent="OnStop",
        session_id=session_id,
    )


if __name__ == "__main__":
    """
    Entry point for running the Chainlit application.
    
    To run the application:
        chainlit run chatbot_ui/app.py
    """
    pass
