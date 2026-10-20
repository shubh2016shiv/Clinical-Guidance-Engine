"""Low-level OpenAI vector store operations shared by application features."""

from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from src.config import get_settings


class OpenAIVectorStoreAdapter:
    """Keep OpenAI SDK calls out of domain packages."""

    def __init__(self, *, client: Any | None = None, api_key: str | None = None) -> None:
        key = api_key or (get_settings().openai_api_key if client is None else None)
        if client is None and not key:
            raise ValueError("OPENAI_API_KEY is required")
        self._owns_client = client is None
        self.client = client if client is not None else AsyncOpenAI(api_key=key)

    async def list_stores(self) -> list[Any]:
        page = await self.client.vector_stores.list()
        return [store async for store in page]

    async def retrieve_store(self, vector_store_id: str) -> Any:
        return await self.client.vector_stores.retrieve(vector_store_id)

    async def list_files(self, vector_store_id: str) -> list[Any]:
        page = await self.client.vector_stores.files.list(vector_store_id=vector_store_id)
        return [file async for file in page]

    async def create_store(self, name: str) -> Any:
        return await self.client.vector_stores.create(name=name)

    async def upload_and_poll(self, vector_store_id: str, path: Path) -> Any:
        with path.open("rb") as content:
            return await self.client.vector_stores.files.upload_and_poll(
                vector_store_id=vector_store_id, file=content
            )

    async def close(self) -> None:
        if self._owns_client:
            await self.client.close()
