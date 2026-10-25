"""Prepare and verify an OpenAI vector store for clinical guidelines."""

from pathlib import Path
from typing import Any


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
