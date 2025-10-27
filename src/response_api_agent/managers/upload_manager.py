"""
Upload Manager for efficient background uploads to OpenAI Vector Stores.

This module provides a dedicated manager for handling file uploads to vector stores
with batching, rate limiting, and background processing capabilities.
"""

import asyncio
import os
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime, UTC
from io import BytesIO
from pathlib import Path
from openai import OpenAI, AsyncOpenAI

from src.config import get_settings
from src.response_api_agent.managers.exceptions import OpenAIResponsesError
from src.logs import get_component_logger, time_execution


class UploadError(OpenAIResponsesError):
    """Exception raised for errors related to file uploads."""

    def __init__(self, message="An error occurred during file upload"):
        super().__init__(message)


class UploadManager:
    """
    Manages efficient background uploads to OpenAI Vector Stores.

    This class provides optimized uploading of files and text content
    to vector stores with batching, rate limiting, and background processing.
    """

    def __init__(
        self, batch_size: int = 5, rate_limit_delay: float = 1.0, max_retries: int = 3
    ):
        """Initialize the Upload Manager.

        Args:
            batch_size: Number of files to upload per batch.
            rate_limit_delay: Seconds to sleep between batches (for API rate limits).
            max_retries: Maximum number of retries for failed uploads.
        """
        self.settings = get_settings()
        self.client = OpenAI(api_key=self.settings.openai_api_key)
        self.async_client = AsyncOpenAI(api_key=self.settings.openai_api_key)
        self.batch_size = batch_size
        self.rate_limit_delay = rate_limit_delay
        self.max_retries = max_retries
        self.logger = get_component_logger("UploadManager")

        # Track background tasks
        self._tasks: Dict[str, Dict[str, Any]] = {}

    @time_execution("UploadManager", "UploadDirectoryToVectorStore")
    async def upload_directory_to_vector_store(
        self,
        vector_store_id: str,
        directory_path: str,
        file_extensions: List[str] = None,
    ) -> str:
        """
        Upload all files with specified extensions from a directory to a vector store.

        Args:
            vector_store_id: Vector store ID
            directory_path: Path to directory containing files
            file_extensions: List of file extensions to include (e.g., ['.pdf', '.md'])
                            If None, all files will be uploaded

        Returns:
            Task ID for tracking the background upload
        """
        directory_path = Path(directory_path)
        if not directory_path.exists():
            error_msg = f"Directory not found: {directory_path}"
            self.logger.error(
                error_msg,
                component="UploadManager",
                subcomponent="UploadDirectoryToVectorStore",
            )
            raise UploadError(error_msg)

        if not directory_path.is_dir():
            error_msg = f"Path is not a directory: {directory_path}"
            self.logger.error(
                error_msg,
                component="UploadManager",
                subcomponent="UploadDirectoryToVectorStore",
            )
            raise UploadError(error_msg)

        # Default to common document extensions if none provided
        if file_extensions is None:
            file_extensions = [".pdf", ".md", ".txt", ".csv", ".json"]

        # Normalize extensions to lowercase
        file_extensions = [
            ext.lower() if ext.startswith(".") else f".{ext.lower()}"
            for ext in file_extensions
        ]

        # Find all matching files
        file_paths = []
        for file in os.listdir(directory_path):
            file_path = os.path.join(directory_path, file)
            if os.path.isfile(file_path):
                _, ext = os.path.splitext(file.lower())
                if not file_extensions or ext in file_extensions:
                    file_paths.append(file_path)

        if not file_paths:
            self.logger.warning(
                f"No files with extensions {file_extensions} found in {directory_path}",
                component="UploadManager",
                subcomponent="UploadDirectoryToVectorStore",
            )
            return None

        self.logger.info(
            f"Found {len(file_paths)} files to upload from {directory_path}",
            component="UploadManager",
            subcomponent="UploadDirectoryToVectorStore",
            file_count=len(file_paths),
            extensions=file_extensions,
        )

        # Start background upload task
        return await self.upload_files_to_vector_store(vector_store_id, file_paths)

    @time_execution("UploadManager", "UploadFilesToVectorStore")
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
        # Generate a unique task ID
        task_id = str(uuid.uuid4())

        # Create background task
        upload_task = asyncio.create_task(
            self._background_upload_files(task_id, vector_store_id, file_paths)
        )

        # Store task metadata
        self._tasks[task_id] = {
            "task": upload_task,
            "vector_store_id": vector_store_id,
            "file_count": len(file_paths),
            "status": "running",
            "created_at": datetime.now(UTC),
            "stats": {
                "total": len(file_paths),
                "successful": 0,
                "failed": 0,
                "pending": len(file_paths),
                "errors": [],
            },
        }

        self.logger.info(
            f"Started background upload task for {len(file_paths)} files",
            component="UploadManager",
            subcomponent="UploadFilesToVectorStore",
            vector_store_id=vector_store_id,
            task_id=task_id,
        )

        return task_id

    async def _background_upload_files(
        self, task_id: str, vector_store_id: str, file_paths: List[str]
    ) -> None:
        """
        Process file uploads in the background with batching and rate limiting.

        Args:
            task_id: Unique task identifier
            vector_store_id: Vector store ID
            file_paths: List of file paths to upload
        """
        try:
            stats = self._tasks[task_id]["stats"]

            # Upload files in batches
            for i in range(0, len(file_paths), self.batch_size):
                batch = file_paths[i : i + self.batch_size]

                batch_results = await self._process_upload_batch(vector_store_id, batch)

                # Update statistics
                stats["successful"] += batch_results["successful"]
                stats["failed"] += batch_results["failed"]
                stats["pending"] -= (
                    batch_results["successful"] + batch_results["failed"]
                )
                stats["errors"].extend(batch_results["errors"])

                self.logger.info(
                    f"Uploaded batch {i // self.batch_size + 1}/{(len(file_paths) - 1) // self.batch_size + 1}",
                    component="UploadManager",
                    subcomponent="BackgroundUpload",
                    vector_store_id=vector_store_id,
                    task_id=task_id,
                    batch_size=len(batch),
                    successful=batch_results["successful"],
                    failed=batch_results["failed"],
                )

                # Rate limit delay between batches
                if i + self.batch_size < len(file_paths):
                    await asyncio.sleep(self.rate_limit_delay)

            # Update task status
            self._tasks[task_id]["status"] = "completed"
            self._tasks[task_id]["completed_at"] = datetime.now(UTC)

            self.logger.info(
                "Background upload completed",
                component="UploadManager",
                subcomponent="BackgroundUpload",
                vector_store_id=vector_store_id,
                task_id=task_id,
                stats=stats,
            )

        except Exception as e:
            # Update task status on error
            if task_id in self._tasks:
                self._tasks[task_id]["status"] = "failed"
                self._tasks[task_id]["error"] = str(e)
                self._tasks[task_id]["completed_at"] = datetime.now(UTC)

            self.logger.error(
                "Error in background upload",
                component="UploadManager",
                subcomponent="BackgroundUpload",
                vector_store_id=vector_store_id,
                task_id=task_id,
                error=str(e),
            )

    async def _process_upload_batch(
        self, vector_store_id: str, file_paths: List[str]
    ) -> Dict[str, Any]:
        """
        Process a batch of files for upload.

        Args:
            vector_store_id: Vector store ID
            file_paths: List of file paths to upload

        Returns:
            Dictionary with batch upload statistics
        """
        tasks = []
        stats = {"successful": 0, "failed": 0, "errors": []}

        # Create tasks for each file in the batch
        for file_path in file_paths:
            file_name = os.path.basename(file_path)
            tasks.append(self._upload_file(vector_store_id, file_path, file_name))

        # Process all uploads in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                stats["failed"] += 1
                error_info = {
                    "file": os.path.basename(file_paths[i]),
                    "error": str(result),
                }
                stats["errors"].append(error_info)
                self.logger.error(
                    f"Failed to upload file {os.path.basename(file_paths[i])}: {result}",
                    component="UploadManager",
                    subcomponent="ProcessUploadBatch",
                )
            else:  # Success
                stats["successful"] += 1

        return stats

    async def _upload_file(
        self, vector_store_id: str, file_path: str, file_name: str, retry_count: int = 0
    ) -> Optional[str]:
        """
        Upload a file to a vector store with retry logic.

        Args:
            vector_store_id: Vector store ID
            file_path: Path to the file
            file_name: File name
            retry_count: Current retry attempt

        Returns:
            File ID on success, None on failure
        """
        try:
            # Upload file directly from path
            with open(file_path, "rb") as file:
                file_obj = await asyncio.to_thread(
                    self.client.files.create, file=file, purpose="assistants"
                )

            # Attach to vector store
            await asyncio.to_thread(
                self.client.vector_stores.files.create,
                vector_store_id=vector_store_id,
                file_id=file_obj.id,
            )

            self.logger.info(
                f"Uploaded {file_name} to vector store {vector_store_id}",
                component="UploadManager",
                subcomponent="UploadFile",
                file_id=file_obj.id,
            )
            return file_obj.id

        except Exception as e:
            # Implement retry logic for transient errors
            if retry_count < self.max_retries:
                self.logger.warning(
                    f"Retrying upload for {file_name} (attempt {retry_count + 1}/{self.max_retries})",
                    component="UploadManager",
                    subcomponent="UploadFile",
                    error=str(e),
                )
                # Exponential backoff for retries
                await asyncio.sleep(2**retry_count)
                return await self._upload_file(
                    vector_store_id, file_path, file_name, retry_count + 1
                )
            else:
                self.logger.error(
                    f"Failed to upload file {file_name} after {self.max_retries} attempts",
                    component="UploadManager",
                    subcomponent="UploadFile",
                    error=str(e),
                )
                raise UploadError(f"Failed to upload file {file_name}: {str(e)}")

    @time_execution("UploadManager", "UploadTextContent")
    async def upload_text_content_to_vector_store(
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
        try:
            self.logger.info(
                f"Uploading text content as {file_name}",
                component="UploadManager",
                subcomponent="UploadTextContent",
                vector_store_id=vector_store_id,
            )

            # Create BytesIO stream
            file_stream = BytesIO(content.encode("utf-8"))
            file_stream.name = file_name

            # Upload file
            file_obj = await asyncio.to_thread(
                self.client.files.create, file=file_stream, purpose="assistants"
            )

            # Attach to vector store
            await asyncio.to_thread(
                self.client.vector_stores.files.create,
                vector_store_id=vector_store_id,
                file_id=file_obj.id,
            )

            self.logger.info(
                f"Uploaded text as {file_name} to vector store {vector_store_id}",
                component="UploadManager",
                subcomponent="UploadTextContent",
                file_id=file_obj.id,
            )
            return file_obj.id

        except Exception as e:
            self.logger.error(
                f"Failed to upload text content as {file_name}",
                component="UploadManager",
                subcomponent="UploadTextContent",
                error=str(e),
            )
            raise UploadError(f"Failed to upload text content: {str(e)}")

    def get_upload_status(self, task_id: str) -> Dict[str, Any]:
        """
        Get status of a background upload task.

        Args:
            task_id: Task ID

        Returns:
            Dictionary with task status information
        """
        if task_id not in self._tasks:
            return {"status": "not_found", "message": f"Task ID {task_id} not found"}

        task_info = self._tasks[task_id]

        # Calculate progress percentage
        stats = task_info["stats"]
        total = stats["total"]
        completed = stats["successful"] + stats["failed"]
        progress = (completed / total * 100) if total > 0 else 0

        return {
            "task_id": task_id,
            "vector_store_id": task_info["vector_store_id"],
            "status": task_info["status"],
            "created_at": task_info["created_at"],
            "completed_at": task_info.get("completed_at"),
            "progress": f"{progress:.1f}%",
            "stats": {
                "total": stats["total"],
                "successful": stats["successful"],
                "failed": stats["failed"],
                "pending": stats["pending"],
                "error_count": len(stats["errors"]),
            },
        }

    def get_detailed_errors(self, task_id: str) -> List[Dict[str, str]]:
        """
        Get detailed error information for a task.

        Args:
            task_id: Task ID

        Returns:
            List of error details
        """
        if task_id not in self._tasks:
            return []

        return self._tasks[task_id]["stats"]["errors"]

    def cancel_upload_task(self, task_id: str) -> bool:
        """
        Cancel an ongoing upload task.

        Args:
            task_id: Task ID

        Returns:
            True if task was cancelled, False otherwise
        """
        if task_id not in self._tasks:
            return False

        task_info = self._tasks[task_id]

        if task_info["status"] in ("completed", "failed"):
            return False

        # Cancel the task
        task_info["task"].cancel()

        # Update task status
        task_info["status"] = "cancelled"
        task_info["completed_at"] = datetime.now(UTC)

        self.logger.info(
            f"Upload task {task_id} cancelled",
            component="UploadManager",
            subcomponent="CancelUploadTask",
        )

        return True

    def cleanup_old_tasks(self, max_age_hours: int = 24) -> int:
        """
        Clean up old completed tasks to free memory.

        Args:
            max_age_hours: Maximum age of completed tasks to keep in hours

        Returns:
            Number of tasks cleaned up
        """
        now = datetime.now(UTC)
        task_ids_to_remove = []

        for task_id, task_info in self._tasks.items():
            if task_info["status"] in ("completed", "failed", "cancelled"):
                completed_at = task_info.get("completed_at")
                if (
                    completed_at
                    and (now - completed_at).total_seconds() > max_age_hours * 3600
                ):
                    task_ids_to_remove.append(task_id)

        # Remove old tasks
        for task_id in task_ids_to_remove:
            del self._tasks[task_id]

        if task_ids_to_remove:
            self.logger.info(
                f"Cleaned up {len(task_ids_to_remove)} old upload tasks",
                component="UploadManager",
                subcomponent="CleanupOldTasks",
            )

        return len(task_ids_to_remove)
