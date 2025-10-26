"""
Healthcare AI Agent Application Entry Point

Main application module for initializing and running the healthcare AI agent.
Handles knowledge base setup, consultation processing, and resource cleanup.
"""

import asyncio
import sys
from typing import Optional
from src.response_api_agent.asclepius_agent import AsclepiusHealthcareAgent
from src.config import get_settings
from src.logs import get_component_logger
from src.response_api_agent.managers.exceptions import ResponsesAPIError


class HealthcareAgentApplication:
    """
    Main application class for the healthcare AI agent.
    
    Manages the complete lifecycle including initialization, consultation processing,
    and proper cleanup of resources.
    """
    
    def __init__(self):
        """Initialize the application."""
        self.settings = get_settings()
        self.logger = get_component_logger("Application")
        self.agent: Optional[AsclepiusHealthcareAgent] = None
        
    async def initialize(self) -> bool:
        """
        Initialize the healthcare agent and knowledge base.
        
        Returns:
            True if initialization successful, False otherwise.
        """
        try:
            self.logger.info(
                "Starting healthcare agent application",
                component="Application",
                subcomponent="Initialize",
                app_version=self.settings.app_version
            )
            
            # Initialize agent
            self.agent = AsclepiusHealthcareAgent(chat_history_limit=10)
            
            # Setup knowledge base with clinical guidelines
            self.logger.info(
                "Setting up clinical knowledge base",
                component="Application",
                subcomponent="Initialize"
            )
            
            vector_store_id = await self.agent.initialize_knowledge_base()
            
            if vector_store_id:
                self.logger.info(
                    "Knowledge base initialized successfully",
                    component="Application",
                    subcomponent="Initialize",
                    vector_store_id=vector_store_id
                )
                return True
            else:
                self.logger.warning(
                    "Knowledge base initialization incomplete, continuing without clinical guidelines",
                    component="Application",
                    subcomponent="Initialize"
                )
                return True  # Continue without knowledge base
                
        except Exception as e:
            self.logger.error(
                "Failed to initialize application",
                component="Application",
                subcomponent="Initialize",
                error=str(e),
                error_type=type(e).__name__
            )
            return False
    
    async def process_consultation(self, question: str, use_guidelines: bool = True) -> Optional[str]:
        """
        Process a healthcare consultation question.
        
        Args:
            question: The healthcare question to process.
            use_guidelines: Whether to use clinical guidelines in the response.
            
        Returns:
            The consultation response, or None if processing failed.
        """
        if not self.agent:
            self.logger.error(
                "Agent not initialized",
                component="Application",
                subcomponent="ProcessConsultation"
            )
            return None
            
        try:
            self.logger.info(
                "Processing consultation request",
                component="Application",
                subcomponent="ProcessConsultation",
                question_length=len(question),
                use_guidelines=use_guidelines
            )
            
            result = await self.agent.consult(
                query=question,
                use_clinical_guidelines=use_guidelines,
                streaming=self.settings.enable_streaming
            )

            # Handle streaming response
            if self.settings.enable_streaming and "stream_generator" in result:
                stream_generator = result["stream_generator"]
                response_parts = []
                extracted_conversation_id = None
                async for chunk_data in stream_generator:
                    # Each chunk is a dict with "chunk" key
                    chunk = chunk_data.get("chunk", "")
                    response_parts.append(chunk)
                    # Extract conversation_id from first chunk
                    if extracted_conversation_id is None:
                        extracted_conversation_id = chunk_data.get("conversation_id")
                response_content = "".join(response_parts)
                # Override conversation_id with extracted value
                if extracted_conversation_id:
                    conversation_id = extracted_conversation_id
                else:
                    conversation_id = result.get("conversation_id", "")
            else:
                response_content = result.get("content", "")
                conversation_id = result.get("conversation_id", "")
            guidelines_used = result.get("guidelines_used", False)
            
            self.logger.info(
                "Consultation completed successfully",
                component="Application",
                subcomponent="ProcessConsultation",
                conversation_id=conversation_id,
                guidelines_used=guidelines_used,
                response_length=len(response_content)
            )
            
            return response_content
            
        except ResponsesAPIError as e:
            self.logger.error(
                "Consultation processing failed",
                component="Application",
                subcomponent="ProcessConsultation",
                error=str(e),
                error_type="ResponsesAPIError"
            )
            return None
            
        except Exception as e:
            self.logger.error(
                "Unexpected error during consultation",
                component="Application",
                subcomponent="ProcessConsultation",
                error=str(e),
                error_type=type(e).__name__
            )
            return None
    
    async def shutdown(self):
        """
        Perform cleanup operations before application shutdown.
        """
        try:
            if self.agent:
                if self.settings.enable_cleanup:
                    self.logger.info(
                        "Shutting down application with cleanup",
                        component="Application",
                        subcomponent="Shutdown"
                    )
                    
                    await self.agent.close_session()
                    
                    self.logger.info(
                        "Application shutdown complete with cleanup",
                        component="Application",
                        subcomponent="Shutdown"
                    )
                else:
                    self.logger.info(
                        "Shutting down application without cleanup (resources will persist)",
                        component="Application",
                        subcomponent="Shutdown"
                    )
        except Exception as e:
            self.logger.error(
                "Error during shutdown",
                component="Application",
                subcomponent="Shutdown",
                error=str(e),
                error_type=type(e).__name__
            )


async def main():
    """
    Main application execution function.
    """
    app = HealthcareAgentApplication()
    
    try:
        # Initialize application
        initialized = await app.initialize()
        
        if not initialized:
            print("ERROR: Application initialization failed. Check logs for details.")
            return 1
        
        print("\n" + "="*80)
        print("Healthcare AI Agent - Consultation Test")
        print("="*80 + "\n")
        
        # Test question for consultation
        test_question = "What are the recommended treatments for type 2 diabetes? Please include information about common medications, their mechanisms of action, and potential side effects."
        
        print(f"Question:\n{test_question}\n")
        print("-"*80 + "\n")
        
        # Process consultation
        response = await app.process_consultation(
            question=test_question,
            use_guidelines=True
        )
        
        if response:
            print(f"Response:\n{response}\n")
            print("-"*80 + "\n")
            print("Consultation completed successfully.")
        else:
            print("ERROR: Failed to get consultation response. Check logs for details.")
            return 1
        
        print("\n" + "="*80)
        print("Test completed successfully")
        print("="*80 + "\n")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\nApplication interrupted by user.")
        return 130
        
    except Exception as e:
        print(f"\nERROR: Unexpected application error: {e}")
        return 1
        
    finally:
        # Ensure cleanup happens
        await app.shutdown()


if __name__ == "__main__":
    """Application entry point."""
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        sys.exit(1)
