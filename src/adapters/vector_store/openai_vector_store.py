"""OpenAI hosted vector stores, used by the Responses API file_search tool."""

import asyncio
from pathlib import Path
from typing import Any

from openai import OpenAI

from src.config import get_settings
from src.logs import get_component_logger, time_execution
from src.response_api_agent.managers.exceptions import VectorStoreError
from src.response_api_agent.managers.upload_manager import UploadManager

GUIDELINES_STORE_NAME = "guidelines"
GUIDELINE_EXTENSIONS = [".pdf", ".md"]


class VectorStoreManager:
    """Create, look up and delete OpenAI vector stores, and load the clinical guidelines."""

    def __init__(self, batch_size: int = 5, rate_limit_delay: float = 1.0):
        self.settings = get_settings()
        self.client = OpenAI(api_key=self.settings.openai_api_key)
        self.batch_size = batch_size
        self.logger = get_component_logger("VectorStore")
        self.upload_manager = UploadManager(
            batch_size=batch_size, rate_limit_delay=rate_limit_delay
        )

    @time_execution("VectorStore", "GetExistingGuidelinesVectorStore")
    async def _get_existing_guidelines_vector_store(self) -> str | None:
        """Return the ID of a guidelines vector store that already has files, else None."""
        try:
            for store in await self.list_vector_stores():
                if store["name"] != GUIDELINES_STORE_NAME or store.get("status") != "completed":
                    continue
                files = await self.get_vector_store_files(store["id"])
                if files and all(file.get("status") == "completed" for file in files):
                    return store["id"]
        except Exception as exc:
            self.logger.error("Error checking for existing guidelines vector store", error=str(exc))
            raise
        return None

    @time_execution("VectorStore", "CreateGuidelinesVectorStore")
    async def create_guidelines_vector_store(self) -> str:
        """Return a guidelines vector store ID, reusing one that already has files.

        A new store expires after a day of inactivity, and the guideline files
        are uploaded to it in the background.

        Raises:
            VectorStoreError: If the guidelines directory is missing or creation fails.
        """
        try:
            existing_id = await self._get_existing_guidelines_vector_store()
            if existing_id:
                self.logger.info(
                    "Reusing existing guidelines vector store", vector_store_id=existing_id
                )
                return existing_id

            guidelines_dir = self.get_clinical_guidelines_directory_path()
            vector_store = await asyncio.to_thread(
                self.client.vector_stores.create,
                name=GUIDELINES_STORE_NAME,
                expires_after={"anchor": "last_active_at", "days": 1},
            )
            upload_task_id = await self.upload_manager.upload_directory_to_vector_store(
                vector_store_id=vector_store.id,
                directory_path=guidelines_dir,
                file_extensions=GUIDELINE_EXTENSIONS,
            )
            await self.upload_manager.wait_for_upload(upload_task_id)
            self.logger.info(
                "Created guidelines vector store",
                vector_store_id=vector_store.id,
                upload_task_id=upload_task_id,
            )
            return vector_store.id

        except Exception as exc:
            self.logger.error("Failed to create guidelines vector store", error=str(exc))
            raise VectorStoreError(f"Failed to create guidelines vector store: {exc}") from exc

    @time_execution("VectorStore", "GetVectorStore")
    async def get_vector_store(self, vector_store_id: str) -> dict[str, Any] | None:
        """Return the vector store's details, or None if it cannot be retrieved."""
        try:
            vector_store = await asyncio.to_thread(
                self.client.vector_stores.retrieve, vector_store_id=vector_store_id
            )
            return {
                "id": vector_store.id,
                "name": vector_store.name,
                "status": vector_store.status,
                "file_counts": vector_store.file_counts,
                "created_at": vector_store.created_at,
                "expires_after": vector_store.expires_after,
            }
        except Exception as exc:
            self.logger.error(
                "Error getting vector store", vector_store_id=vector_store_id, error=str(exc)
            )
            return None

    @time_execution("VectorStore", "ListVectorStores")
    async def list_vector_stores(self) -> list[dict[str, Any]]:
        """Return id, name, status, file counts and creation time of every vector store.

        Raises:
            VectorStoreError: If the listing fails.
        """
        try:
            vector_stores = await asyncio.to_thread(self.client.vector_stores.list)
            return [
                {
                    "id": store.id,
                    "name": store.name,
                    "status": store.status,
                    "file_counts": store.file_counts,
                    "created_at": store.created_at,
                }
                for store in vector_stores.data
            ]
        except Exception as exc:
            self.logger.error("Error listing vector stores", error=str(exc))
            raise VectorStoreError(f"Failed to list vector stores: {exc}") from exc

    @time_execution("VectorStore", "DeleteVectorStore")
    async def delete_vector_store(self, vector_store_id: str) -> bool:
        """Delete the vector store. Returns False if the deletion fails."""
        try:
            await asyncio.to_thread(
                self.client.vector_stores.delete, vector_store_id=vector_store_id
            )
            return True
        except Exception as exc:
            self.logger.error(
                "Error deleting vector store", vector_store_id=vector_store_id, error=str(exc)
            )
            return False

    @time_execution("VectorStore", "GetVectorStoreFiles")
    async def get_vector_store_files(self, vector_store_id: str) -> list[dict[str, Any]]:
        """Return id, status and creation time of each file in the vector store.

        Includes files that are not yet in the "completed" state.

        Raises:
            VectorStoreError: If the listing fails.
        """
        try:
            files = await asyncio.to_thread(
                self.client.vector_stores.files.list, vector_store_id=vector_store_id
            )
            return [
                {"id": file.id, "status": file.status, "created_at": file.created_at}
                for file in files.data
            ]
        except Exception as exc:
            self.logger.error(
                "Error getting vector store files",
                vector_store_id=vector_store_id,
                error=str(exc),
            )
            raise VectorStoreError(f"Failed to get vector store files: {exc}") from exc

    def get_clinical_guidelines_directory_path(self) -> str:
        """Return the absolute path of the configured clinical guidelines directory.

        Raises:
            VectorStoreError: If the directory does not exist.
        """
        project_root = Path(__file__).resolve().parents[3]
        guidelines_path = project_root / self.settings.clinical_guidelines_directory
        if not guidelines_path.exists():
            raise VectorStoreError(f"Clinical guidelines directory not found at: {guidelines_path}")
        return str(guidelines_path)
