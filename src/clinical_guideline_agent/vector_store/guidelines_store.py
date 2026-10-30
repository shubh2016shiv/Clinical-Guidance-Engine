"""Prepare and verify an OpenAI vector store for clinical guidelines."""

from pathlib import Path
from typing import Any

from src.adapters.vector_store.exceptions import VectorDBCollectionError

from ..schema import IndexedGuidelineFile


class GuidelinesStore:
    """Manage only stores used by this component; never delete shared stores."""

    def __init__(
        self, openai_vector_store_client: Any, *, store_name: str = "clinical_guidelines"
    ) -> None:
        self.openai_vector_store_client = openai_vector_store_client
        self.store_name = store_name

    async def check_ready(self, vector_store_id: str) -> None:
        """Require a completed store with at least one completed file."""
        vector_store = await self.openai_vector_store_client.retrieve_vector_store(vector_store_id)
        if vector_store.status != "completed":
            raise ValueError(f"Vector store {vector_store_id} is {vector_store.status}")

        vector_store_files = await self.openai_vector_store_client.list_vector_store_files(
            vector_store_id
        )
        for vector_store_file in vector_store_files:
            if vector_store_file.status != "completed":
                raise ValueError(
                    f"Guideline file {vector_store_file.id} is {vector_store_file.status}"
                )
        if not vector_store_files:
            raise ValueError(f"Vector store {vector_store_id} has no guideline files")

    async def start_fresh(self, previous_vector_store_id: str | None = None) -> str:
        """Delete the previous store and return a new empty one's ID.

        The chat UI uploads guidelines itself, so each run starts with nothing
        indexed rather than reusing whatever a previous run left behind.
        """
        # A known ID is deleted directly because listing is eventually
        # consistent: a store created moments ago may not appear in the list
        # yet, which would leak it instead of replacing it.
        if previous_vector_store_id:
            await self.openai_vector_store_client.delete_vector_store(previous_vector_store_id)
        # Listing still catches stores left behind by earlier runs, whose IDs
        # this process never saw.
        for vector_store in await self.openai_vector_store_client.list_vector_stores():
            if vector_store.name == self.store_name and vector_store.id != previous_vector_store_id:
                await self.openai_vector_store_client.delete_vector_store(vector_store.id)
        vector_store = await self.openai_vector_store_client.create_vector_store(self.store_name)
        return vector_store.id

    async def add_files(
        self,
        vector_store_id: str,
        file_paths: list[Path],
        *,
        display_names: dict[Path, str] | None = None,
    ) -> Any:
        """Upload files a user attached in the chat into an existing store."""
        if not file_paths:
            raise ValueError("No guideline files to upload")
        file_batch = await self.openai_vector_store_client.upload_files_and_wait(
            vector_store_id, file_paths, display_names=display_names
        )
        if file_batch.file_counts.failed:
            raise RuntimeError(
                f"{file_batch.file_counts.failed} of {file_batch.file_counts.total} "
                f"guideline uploads failed for vector store {vector_store_id}"
            )
        return file_batch

    async def list_files(self, vector_store_id: str) -> list[IndexedGuidelineFile]:
        """List the files indexed in a store, using the names they were uploaded with."""
        client = self.openai_vector_store_client
        vector_store_files = await client.list_vector_store_files(vector_store_id)

        indexed_files: list[IndexedGuidelineFile] = []
        for vector_store_file in vector_store_files:
            try:
                file_info = await client.retrieve_file_info(vector_store_file.id)
            except VectorDBCollectionError:
                # A file that was just removed can still show up in the store's list
                # for a moment, although the upload itself is already deleted. It is
                # gone, so leave it out rather than fail the whole list.
                continue
            indexed_files.append(
                IndexedGuidelineFile(
                    file_id=vector_store_file.id,
                    filename=file_info.filename,
                    status=vector_store_file.status,
                    size_bytes=file_info.size_bytes,
                )
            )
        return indexed_files

    async def remove_file(self, vector_store_id: str, file_id: str) -> None:
        """Remove one file from a store, and delete the uploaded file as well."""
        await self.openai_vector_store_client.remove_file(vector_store_id, file_id)

    async def find_ready_store(self) -> str | None:
        """Reuse a ready store with this name when no ID was supplied."""
        for vector_store in await self.openai_vector_store_client.list_vector_stores():
            if vector_store.name != self.store_name:
                continue
            try:
                await self.check_ready(vector_store.id)
                return vector_store.id
            except ValueError:
                continue
        return None

    async def ensure_ready(
        self, directory: str | Path, *, vector_store_id: str | None = None
    ) -> str:
        """Verify a known store, reuse a ready one, or upload local files."""
        if vector_store_id:
            await self.check_ready(vector_store_id)
            return vector_store_id

        existing_vector_store_id = await self.find_ready_store()
        if existing_vector_store_id:
            return existing_vector_store_id

        guidelines_directory = Path(directory)
        if not guidelines_directory.is_dir():
            raise ValueError(f"Guideline directory does not exist: {guidelines_directory}")
        guideline_files = sorted(
            guideline_file_path
            for guideline_file_path in guidelines_directory.iterdir()
            if guideline_file_path.is_file()
            and guideline_file_path.suffix.lower() in {".pdf", ".md"}
        )
        if not guideline_files:
            raise ValueError(f"No PDF or Markdown guidelines found in {guidelines_directory}")

        vector_store = await self.openai_vector_store_client.create_vector_store(self.store_name)
        file_batch = await self.openai_vector_store_client.upload_files_and_wait(
            vector_store.id, guideline_files
        )
        if file_batch.file_counts.failed:
            raise RuntimeError(
                f"{file_batch.file_counts.failed} of {file_batch.file_counts.total} "
                f"guideline uploads failed for vector store {vector_store.id}"
            )
        await self.check_ready(vector_store.id)
        return vector_store.id
