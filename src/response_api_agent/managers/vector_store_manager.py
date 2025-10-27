import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, UTC
from openai import OpenAI, AsyncOpenAI
from pathlib import Path

from src.config import get_settings
from src.response_api_agent.managers.exceptions import VectorStoreError
from src.response_api_agent.managers.upload_manager import UploadManager
from src.logs import get_component_logger, time_execution


class VectorStoreManager:
    """
    Manages OpenAI Vector Stores for the Responses API.

    Vector stores are used with the file_search tool to enable
    semantic search over uploaded documents.
    """

    def __init__(self, batch_size: int = 5, rate_limit_delay: float = 1.0):
        """Initialize the Vector Store Manager.

        Args:
            batch_size: Number of files to upload per batch.
            rate_limit_delay: Seconds to sleep between batches (for API rate limits).
        """
        self.settings = get_settings()
        self.client = OpenAI(api_key=self.settings.openai_api_key)
        self.async_client = AsyncOpenAI(api_key=self.settings.openai_api_key)
        self._vector_store_cache: Dict[str, Dict[str, Any]] = {}
        self.batch_size = batch_size
        self.rate_limit_delay = rate_limit_delay
        self.logger = get_component_logger("VectorStore")

        # Initialize the upload manager with the same configuration
        self.upload_manager = UploadManager(
            batch_size=batch_size, rate_limit_delay=rate_limit_delay
        )

    @time_execution("VectorStore", "GetExistingGuidelinesVectorStore")
    async def _get_existing_guidelines_vector_store(self) -> Optional[str]:
        """
        Check for existing guidelines vector store with files.

        Returns:
            Vector store ID if found with files, None otherwise.
        """
        try:
            self.logger.info(
                "Checking for existing guidelines vector store",
                component="VectorStore",
                subcomponent="GetExistingGuidelinesVectorStore",
            )

            # List all vector stores
            vector_stores = await self.list_vector_stores()

            # Find vector store with name "guidelines" that has files
            for vs in vector_stores:
                if vs["name"] == "guidelines":
                    # Check if vector store has files
                    try:
                        files = await self.get_vector_store_files(vs["id"])
                        if files:  # Has files
                            self.logger.info(
                                "Found existing guidelines vector store with files",
                                component="VectorStore",
                                subcomponent="GetExistingGuidelinesVectorStore",
                                vector_store_id=vs["id"],
                                file_count=len(files),
                            )
                            return vs["id"]
                        else:
                            self.logger.info(
                                "Found guidelines vector store but no files",
                                component="VectorStore",
                                subcomponent="GetExistingGuidelinesVectorStore",
                                vector_store_id=vs["id"],
                            )
                    except Exception as e:
                        self.logger.warning(
                            "Error checking files for vector store",
                            component="VectorStore",
                            subcomponent="GetExistingGuidelinesVectorStore",
                            vector_store_id=vs["id"],
                            error=str(e),
                        )
                        continue

            self.logger.info(
                "No existing guidelines vector store with files found",
                component="VectorStore",
                subcomponent="GetExistingGuidelinesVectorStore",
            )
            return None

        except Exception as e:
            self.logger.error(
                "Error checking for existing guidelines vector store",
                component="VectorStore",
                subcomponent="GetExistingGuidelinesVectorStore",
                error=str(e),
            )
            return None

    @time_execution("VectorStore", "CreateGuidelinesVectorStore")
    async def create_guidelines_vector_store(self) -> str:
        """
        Create a vector store for medical guidelines and start background upload.
        Reuses existing vector store if one with files already exists.

        Returns:
            Vector store ID (upload happens asynchronously in background).
        """
        try:
            self.logger.info(
                "Creating guidelines vector store",
                component="VectorStore",
                subcomponent="CreateGuidelinesVectorStore",
            )

            # Check for existing vector store with files first
            existing_vector_store_id = (
                await self._get_existing_guidelines_vector_store()
            )
            if existing_vector_store_id:
                self.logger.info(
                    "Reusing existing guidelines vector store",
                    component="VectorStore",
                    subcomponent="CreateGuidelinesVectorStore",
                    vector_store_id=existing_vector_store_id,
                    status="reused",
                )
                # Cache the existing vector store for consistency
                if existing_vector_store_id not in self._vector_store_cache:
                    self._vector_store_cache[existing_vector_store_id] = {
                        "name": "clinical_guidelines",
                        "created_at": datetime.now(UTC),
                        "status": "reused",
                    }
                return existing_vector_store_id

            # Create vector store with 24 hour expiry
            vector_store = await asyncio.to_thread(
                self.client.vector_stores.create,
                name="guidelines",
                expires_after={"anchor": "last_active_at", "days": 1},
            )

            self.logger.info(
                "Created guidelines vector store",
                component="VectorStore",
                subcomponent="CreateGuidelinesVectorStore",
                vector_store_id=vector_store.id,
                status="created",
            )

            # Start background upload using UploadManager (fire-and-forget)
            guideline_dir = self.get_clinical_guidelines_directory_path()
            upload_task_id = await self.upload_manager.upload_directory_to_vector_store(
                vector_store_id=vector_store.id,
                directory_path=guideline_dir,
                file_extensions=[".pdf", ".md"],
            )

            # Cache the full vector store object with upload task tracking
            self._vector_store_cache[vector_store.id] = {
                "name": "clinical_guidelines",
                "object": vector_store.model_dump(),  # Full serialized object for easy access
                "created_at": datetime.now(UTC),
                "status": "created",
                "upload_task_id": upload_task_id,  # Track upload task for status monitoring
            }

            self.logger.info(
                "Started background upload task for guidelines",
                component="VectorStore",
                subcomponent="CreateGuidelinesVectorStore",
                vector_store_id=vector_store.id,
                upload_task_id=upload_task_id,
            )

            return vector_store.id

        except Exception as e:
            self.logger.error(
                "Failed to create guidelines vector store",
                component="VectorStore",
                subcomponent="CreateGuidelinesVectorStore",
                error=str(e),
            )
            raise VectorStoreError(
                f"Failed to create guidelines vector store: {str(e)}"
            )

    # async def _upload_guideline_files(self, vector_store_id: str) -> None:
    #     """
    #     Upload guideline files (PDFs and MDs) from local folder to vector store in background.

    #     Args:
    #         vector_store_id: Vector store ID
    #     """
    #     try:
    #         guideline_dir = os.path.join(os.getcwd(), "clinical_guidelines")

    #         if not os.path.exists(guideline_dir):
    #             self.logger.warning(f"Guidelines directory not found: {guideline_dir}")
    #             return

    #         # Support both PDF and MD files
    #         guideline_files = [
    #             f for f in os.listdir(guideline_dir)
    #             if f.lower().endswith(('.pdf', '.md'))
    #         ]

    #         if not guideline_files:
    #             self.logger.warning("No supported files (PDF/MD) found in guidelines directory")
    #             return

    #         self.logger.info(f"Found {len(guideline_files)} guideline files (PDF/MD) to upload")

    #         stats = {"total": len(guideline_files), "successful": 0, "failed": 0, "errors": []}

    #         # Upload files in batches
    #         for i in range(0, len(guideline_files), self.batch_size):
    #             batch = guideline_files[i:i + self.batch_size]
    #             tasks = []

    #             for filename in batch:
    #                 file_path = os.path.join(guideline_dir, filename)
    #                 tasks.append(
    #                     self._upload_file_to_vector_store(vector_store_id, file_path, filename)
    #                 )

    #             results = await asyncio.gather(*tasks, return_exceptions=True)

    #             # Process results
    #             for result in results:
    #                 if isinstance(result, Exception):
    #                     stats["failed"] += 1
    #                     stats["errors"].append(str(result))
    #                     self.logger.error(f"Batch upload error: {result}")
    #                 else:  # Assume result is file_id on success
    #                     stats["successful"] += 1

    #             self.logger.info(f"Uploaded batch {i // self.batch_size + 1} ({len(batch)} files)")

    #             # Rate limit delay between batches
    #             if i + self.batch_size < len(guideline_files):
    #                 await asyncio.sleep(self.rate_limit_delay)

    #         self.logger.info(f"Background upload completed for guideline files: {stats}")

    #     except Exception as e:
    #         self.logger.error(f"Error in background upload of guideline files: {e}")
    #         # Don't raise here as this is a background task

    # async def _upload_file_to_vector_store(
    #     self, vector_store_id: str, file_path: str, file_name: str
    # ) -> Optional[str]:
    #     """
    #     Upload a file to a vector store.

    #     Args:
    #         vector_store_id: Vector store ID
    #         file_path: Path to the file
    #         file_name: File name

    #     Returns:
    #         File ID on success, None on failure.
    #     """
    #     try:
    #         # Upload file directly from path (simpler than BytesIO)
    #         with open(file_path, "rb") as file:
    #             file_obj = await asyncio.to_thread(
    #                 self.client.files.create,
    #                 file=file,
    #                 purpose="assistants"
    #             )

    #         await asyncio.to_thread(
    #             self.client.vector_stores.files.create,
    #             vector_store_id=vector_store_id,
    #             file_id=file_obj.id
    #         )

    #         self.logger.info(f"Uploaded {file_name} to vector store {vector_store_id}")
    #         return file_obj.id

    #     except Exception as e:
    #         self.logger.error(f"Failed to upload file {file_name}: {e}")
    #         return None

    # async def _upload_text_to_vector_store(
    #     self, vector_store_id: str, content: str, file_name: str
    # ) -> str:
    #     """
    #     Upload text content to a vector store.

    #     Args:
    #         vector_store_id: Vector store ID
    #         content: Text content
    #         file_name: File name

    #     Returns:
    #         File ID
    #     """
    #     try:
    #         # Create BytesIO stream
    #         file_stream = BytesIO(content.encode("utf-8"))
    #         file_stream.name = file_name

    #         # Upload file
    #         file_obj = await asyncio.to_thread(
    #             self.client.files.create,
    #             file=file_stream,
    #             purpose="assistants"
    #         )

    #         # Attach to vector store
    #         await asyncio.to_thread(
    #             self.client.vector_stores.files.create,
    #             vector_store_id=vector_store_id,
    #             file_id=file_obj.id
    #         )

    #         self.logger.info(f"Uploaded text as {file_name} to vector store {vector_store_id}")
    #         return file_obj.id

    #     except Exception as e:
    #         self.logger.error(f"Failed to upload text content: {e}")
    #         raise VectorStoreError(f"Failed to upload text content: {str(e)}")

    @time_execution("VectorStore", "GetVectorStore")
    async def get_vector_store(self, vector_store_id: str) -> Optional[Dict[str, Any]]:
        """
        Get vector store information.

        Args:
            vector_store_id: Vector store ID

        Returns:
            Vector store information or None
        """
        try:
            self.logger.info(
                "Retrieving vector store",
                component="VectorStore",
                subcomponent="GetVectorStore",
                vector_store_id=vector_store_id,
            )

            vector_store = await asyncio.to_thread(
                self.client.vector_stores.retrieve, vector_store_id=vector_store_id
            )

            self.logger.info(
                "Retrieved vector store successfully",
                component="VectorStore",
                subcomponent="GetVectorStore",
                vector_store_id=vector_store_id,
                status=vector_store.status,
            )

            return {
                "id": vector_store.id,
                "name": vector_store.name,
                "status": vector_store.status,
                "file_counts": vector_store.file_counts,
                "created_at": vector_store.created_at,
                "expires_after": vector_store.expires_after,
            }

        except Exception as e:
            self.logger.error(
                "Error getting vector store",
                component="VectorStore",
                subcomponent="GetVectorStore",
                vector_store_id=vector_store_id,
                error=str(e),
            )
            return None

    @time_execution("VectorStore", "ListVectorStores")
    async def list_vector_stores(self) -> List[Dict[str, Any]]:
        """
        List all vector stores.

        Returns:
            List of vector store information
        """
        try:
            self.logger.info(
                "Listing all vector stores",
                component="VectorStore",
                subcomponent="ListVectorStores",
            )

            vector_stores = await asyncio.to_thread(self.client.vector_stores.list)

            result = []
            for vs in vector_stores.data:
                result.append(
                    {
                        "id": vs.id,
                        "name": vs.name,
                        "status": vs.status,
                        "file_counts": vs.file_counts,
                        "created_at": vs.created_at,
                    }
                )

            self.logger.info(
                "Vector stores listed successfully",
                component="VectorStore",
                subcomponent="ListVectorStores",
                count=len(result),
            )

            return result

        except Exception as e:
            self.logger.error(
                "Error listing vector stores",
                component="VectorStore",
                subcomponent="ListVectorStores",
                error=str(e),
            )
            raise VectorStoreError(f"Failed to list vector stores: {str(e)}")

    @time_execution("VectorStore", "DeleteVectorStore")
    async def delete_vector_store(self, vector_store_id: str) -> bool:
        """
        Delete a vector store.

        Args:
            vector_store_id: Vector store ID

        Returns:
            True if successful
        """
        try:
            self.logger.info(
                "Deleting vector store",
                component="VectorStore",
                subcomponent="DeleteVectorStore",
                vector_store_id=vector_store_id,
            )

            await asyncio.to_thread(
                self.client.vector_stores.delete, vector_store_id=vector_store_id
            )

            # Remove from cache
            if vector_store_id in self._vector_store_cache:
                del self._vector_store_cache[vector_store_id]

            self.logger.info(
                "Vector store deleted successfully",
                component="VectorStore",
                subcomponent="DeleteVectorStore",
                vector_store_id=vector_store_id,
            )
            return True

        except Exception as e:
            self.logger.error(
                "Error deleting vector store",
                component="VectorStore",
                subcomponent="DeleteVectorStore",
                vector_store_id=vector_store_id,
                error=str(e),
            )
            return False

    @time_execution("VectorStore", "GetVectorStoreFiles")
    async def get_vector_store_files(
        self, vector_store_id: str
    ) -> List[Dict[str, Any]]:
        """
        Get files in a vector store.

        Args:
            vector_store_id: Vector store ID

        Returns:
            List of file information
        """
        try:
            self.logger.info(
                "Getting files for vector store",
                component="VectorStore",
                subcomponent="GetVectorStoreFiles",
                vector_store_id=vector_store_id,
            )

            files = await asyncio.to_thread(
                self.client.vector_stores.files.list, vector_store_id=vector_store_id
            )

            result = []
            for file in files.data:
                result.append(
                    {
                        "id": file.id,
                        "status": file.status,
                        "created_at": file.created_at,
                    }
                )

            self.logger.info(
                "Retrieved vector store files successfully",
                component="VectorStore",
                subcomponent="GetVectorStoreFiles",
                vector_store_id=vector_store_id,
                file_count=len(result),
            )

            return result  # TODO: Send the result after filtering out files that are not in the "completed" state

        except Exception as e:
            self.logger.error(
                "Error getting vector store files",
                component="VectorStore",
                subcomponent="GetVectorStoreFiles",
                vector_store_id=vector_store_id,
                error=str(e),
            )
            raise VectorStoreError(f"Failed to get vector store files: {str(e)}")

    def clear_cache(self) -> None:
        """Clear the vector store cache."""
        self._vector_store_cache.clear()
        self.logger.info(
            "Vector store cache cleared",
            component="VectorStore",
            subcomponent="ClearCache",
        )

    # Upload delegation methods for unified public API
    async def upload_files_to_vector_store(
        self, vector_store_id: str, file_paths: List[str]
    ) -> str:
        """
        Upload multiple files to a vector store in the background.

        Args:
            vector_store_id: Vector store ID
            file_paths: List of file paths to upload

        Returns:
            Task ID for tracking the background upload
        """
        return await self.upload_manager.upload_files_to_vector_store(
            vector_store_id, file_paths
        )

    async def upload_text_to_vector_store(
        self, vector_store_id: str, content: str, file_name: str
    ) -> str:
        """
        Upload text content to a vector store.

        Args:
            vector_store_id: Vector store ID
            content: Text content
            file_name: File name

        Returns:
            File ID
        """
        return await self.upload_manager.upload_text_content_to_vector_store(
            vector_store_id, content, file_name
        )

    def get_upload_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        Get status of a background upload task.

        Args:
            task_id: Task ID

        Returns:
            Dictionary with task status information
        """
        return self.upload_manager.get_upload_status(task_id)

    def cancel_upload_task(self, task_id: str) -> bool:
        """
        Cancel an ongoing upload task.

        Args:
            task_id: Task ID

        Returns:
            True if task was cancelled, False otherwise
        """
        return self.upload_manager.cancel_upload_task(task_id)

    def get_upload_detailed_errors(self, task_id: str) -> List[Dict[str, str]]:
        """
        Get detailed error information for a task.

        Args:
            task_id: Task ID

        Returns:
            List of error details
        """
        return self.upload_manager.get_detailed_errors(task_id)

    def cleanup_old_upload_tasks(self, max_age_hours: int = 24) -> int:
        """
        Clean up old completed upload tasks to free memory.

        Args:
            max_age_hours: Maximum age of completed tasks to keep in hours

        Returns:
            Number of tasks cleaned up
        """
        return self.upload_manager.cleanup_old_tasks(max_age_hours)

    def get_clinical_guidelines_directory_path(self) -> str:
        """
        Get the full path to the clinical guidelines directory.

        Returns:
            Full absolute path to the clinical guidelines directory
        """
        # Get project root: 4 levels up from this file (managers/vector_store_manager.py)
        project_root = Path(__file__).parent.parent.parent.parent
        guidelines_path = project_root / self.settings.clinical_guidelines_directory

        if not guidelines_path.exists():
            raise VectorStoreError(
                f"Clinical guidelines directory not found at: {guidelines_path}"
            )

        return str(guidelines_path)

    async def create_guidelines_vector_store_with_upload(
        self, file_extensions: List[str] = None
    ) -> str:
        """
        Create a guidelines vector store and upload all guideline files from the configured directory.

        This is a convenience method that combines vector store creation and file upload
        using the centralized clinical guidelines directory configuration.

        Args:
            file_extensions: List of file extensions to upload (default: ['.pdf', '.md'])

        Returns:
            Vector store ID
        """
        if file_extensions is None:
            file_extensions = [".pdf", ".md"]

        # Create vector store first
        vector_store_id = await self.create_guidelines_vector_store()

        # Get the configured directory path
        guideline_dir = self.get_clinical_guidelines_directory_path()

        # Start upload using UploadManager
        upload_task_id = await self.upload_manager.upload_directory_to_vector_store(
            vector_store_id=vector_store_id,
            directory_path=guideline_dir,
            file_extensions=file_extensions,
        )

        self.logger.info(
            "Started guidelines upload with centralized configuration",
            component="VectorStore",
            subcomponent="CreateGuidelinesWithUpload",
            vector_store_id=vector_store_id,
            upload_task_id=upload_task_id,
            guideline_directory=guideline_dir,
            extensions=file_extensions,
        )

        return vector_store_id
