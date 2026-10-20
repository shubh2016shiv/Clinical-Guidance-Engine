"""Prepare and verify an OpenAI vector store for clinical guidelines."""

from pathlib import Path
from typing import Any


class GuidelinesStore:
    """Manage only stores used by this component; never delete shared stores."""

    def __init__(self, adapter: Any, *, name: str = "clinical_guidelines") -> None:
        self.adapter = adapter
        self.name = name

    async def check_ready(self, vector_store_id: str) -> None:
        """Require a completed store with at least one completed file."""
        store = await self.adapter.retrieve_store(vector_store_id)
        if store.status != "completed":
            raise ValueError(f"Vector store {vector_store_id} is {store.status}")

        files = await self.adapter.list_files(vector_store_id)
        for file in files:
            if file.status != "completed":
                raise ValueError(f"Guideline file {file.id} is {file.status}")
        if not files:
            raise ValueError(f"Vector store {vector_store_id} has no guideline files")

    async def find_ready_store(self) -> str | None:
        """Reuse a ready store with this name when no ID was supplied."""
        for store in await self.adapter.list_stores():
            if store.name != self.name:
                continue
            try:
                await self.check_ready(store.id)
                return store.id
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

        existing_id = await self.find_ready_store()
        if existing_id:
            return existing_id

        path = Path(directory)
        if not path.is_dir():
            raise ValueError(f"Guideline directory does not exist: {path}")
        files = sorted(
            file
            for file in path.iterdir()
            if file.is_file() and file.suffix.lower() in {".pdf", ".md"}
        )
        if not files:
            raise ValueError(f"No PDF or Markdown guidelines found in {path}")

        store = await self.adapter.create_store(self.name)
        for file in files:
            uploaded = await self.adapter.upload_and_poll(store.id, file)
            if uploaded.status != "completed":
                raise RuntimeError(f"Guideline upload failed for {file.name}: {uploaded.status}")
        await self.check_ready(store.id)
        return store.id
