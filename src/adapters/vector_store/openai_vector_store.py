"""Client for OpenAI-hosted vector stores and their files.

Big-picture flow:
1. Open one connection to OpenAI, either reusing a client the caller
   already has, or creating a fresh one from an API key.
2. Offer small, focused methods for the OpenAI vector-store operations the
   clinical guideline agent needs: list stores, retrieve one, list its
   files, create a new store, upload a batch of files into it with
   progress logging, and delete a store.
3. Translate every OpenAI SDK failure into this adapter's own error types
   (see exceptions.py), so callers can catch ``VectorDBProviderError`` and
   never need to know OpenAI-specific exception classes.
4. Only close the underlying OpenAI client if THIS object is the one that
   created it, so a shared client passed in by the caller is not closed
   out from under them.
"""

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, TypeVar

import openai
from openai import AsyncOpenAI

from src.config import get_settings
from src.logs import get_logger

from .exceptions import (
    VectorDBAuthenticationError,
    VectorDBCollectionError,
    VectorDBConfigError,
    VectorDBConnectionError,
    VectorDBProviderError,
    VectorDBRateLimitError,
)

logger = get_logger(__name__)

T = TypeVar("T")
# T: a placeholder type so _run below can wrap an operation returning
# anything (a list, one store object, a bool) while keeping that exact
# return type visible to type checkers.


# _map_openai_error: turns whatever the OpenAI SDK raised into this
# adapter's matching error type. Needed so every method below gets the
# same translation from one place, instead of repeating the same checks.
def _map_openai_error(exc: Exception, operation: str) -> VectorDBProviderError:
    """Return the adapter error matching ``exc``, labelled with ``operation``."""
    message = f"Failed to {operation}: {exc}"
    # "isinstance(exc, SomeClass)" asks "is this error specifically a
    # SomeClass (or a subclass)?" - safer than matching on the error text,
    # which can change between SDK versions. The most specific types are
    # checked first, because several of them share a common parent.
    if isinstance(exc, openai.AuthenticationError | openai.PermissionDeniedError):
        # The API key was rejected, or it lacks permission for this call.
        return VectorDBAuthenticationError(message)
    if isinstance(exc, openai.RateLimitError):
        # Too many requests were sent too quickly.
        return VectorDBRateLimitError(message)
    if isinstance(exc, openai.NotFoundError):
        # The store or file does not exist. A hosted vector store is this
        # adapter's equivalent of a collection, so it maps to the
        # collection error rather than a generic failure.
        return VectorDBCollectionError(message)
    if isinstance(exc, openai.APIConnectionError):
        # The request never reached OpenAI, or timed out on the way. This
        # also covers calls made after close(), when the underlying HTTP
        # client is already shut down.
        return VectorDBConnectionError(message)
    if isinstance(exc, OSError):
        # A local file could not be read (missing, or no permission).
        # FileNotFoundError and PermissionError are both OSError subclasses.
        # This is a problem with what was configured, not with OpenAI, so
        # it is reported as a configuration error.
        return VectorDBConfigError(message)
    # Anything else unexpected still becomes an adapter error, so callers
    # only ever have to catch VectorDBProviderError.
    return VectorDBProviderError(message)


# OpenAIVectorStoreClient: a thin wrapper around OpenAI's own vector-store
# SDK calls. Needed so the guideline store manager (elsewhere in the app)
# can work with simple, named methods instead of calling the OpenAI SDK
# directly everywhere it needs a vector store.
class OpenAIVectorStoreClient:
    """Expose OpenAI vector-store SDK calls to the guideline store manager."""

    def __init__(
        self, *, openai_client: Any | None = None, openai_api_key: str | None = None
    ) -> None:
        # The "*" makes both arguments keyword-only, so callers must write
        # them by name rather than relying on position.
        resolved_openai_api_key = openai_api_key or (
            get_settings().openai_api_key if openai_client is None else None
        )
        # Step 1: work out which API key to use. This is a conditional
        # expression (a compact "if/else" that produces a value): if an
        # "openai_client" was already given, there is no need for a key at
        # all (we use None); otherwise, fall back to the app's configured
        # key when none was passed in directly.
        if openai_client is None and not resolved_openai_api_key:
            # Step 2: if we are expected to build our OWN client but have
            # no key to build it with, fail clearly right away.
            raise ValueError("OPENAI_API_KEY is required")
        self._owns_openai_client = openai_client is None
        # self._owns_openai_client: remembers whether THIS object created
        # the OpenAI client itself (True) or was simply handed one to
        # reuse (False) - used by close() below to decide whether it is
        # this object's responsibility to shut the client down.
        self.openai_client = (
            openai_client
            if openai_client is not None
            else AsyncOpenAI(api_key=resolved_openai_api_key)
        )
        # self.openai_client: either the caller's existing client, or a
        # brand-new one built from the resolved API key.

    async def _run(self, operation: str, request: Callable[[], Awaitable[T]]) -> T:
        """Run one OpenAI call and convert any failure to an adapter error."""
        # "request" is a function taking no arguments that performs the
        # actual SDK call when awaited. Passing a function (rather than
        # calling it directly) lets this helper wrap the call in one shared
        # try/except without needing to know what the operation does.
        try:
            return await request()
        except VectorDBProviderError:
            # Already one of our own errors - let it through unchanged
            # rather than wrapping it a second time.
            raise
        except Exception as exc:
            # "from exc" keeps the original SDK error attached as the
            # cause, so tracebacks still show what really went wrong.
            raise _map_openai_error(exc, operation) from exc

    async def list_vector_stores(self) -> list[Any]:
        """Return OpenAI SDK vector-store objects."""

        async def read() -> list[Any]:
            vector_store_page = await self.openai_client.vector_stores.list()
            # OpenAI returns results one "page" at a time for large lists;
            # "async for ... in vector_store_page" below walks through every
            # item across all pages automatically.
            return [vector_store async for vector_store in vector_store_page]
            # This is an "async list comprehension": like a normal list
            # comprehension, but built by looping over an async source one
            # item at a time instead of a plain in-memory list.

        return await self._run("list vector stores", read)

    async def retrieve_vector_store(self, vector_store_id: str) -> Any:
        """Return one OpenAI SDK vector-store object."""
        return await self._run(
            f"retrieve vector store {vector_store_id}",
            lambda: self.openai_client.vector_stores.retrieve(vector_store_id),
        )
        # "lambda: ..." is a tiny unnamed function used here because the
        # operation is a single call, so a full "async def" would add noise.

    async def list_vector_store_files(self, vector_store_id: str) -> list[Any]:
        """Return OpenAI SDK file records attached to a vector store."""

        async def read() -> list[Any]:
            vector_store_file_page = await self.openai_client.vector_stores.files.list(
                vector_store_id=vector_store_id
            )
            return [vector_store_file async for vector_store_file in vector_store_file_page]

        return await self._run(f"list files in vector store {vector_store_id}", read)

    async def create_vector_store(self, vector_store_name: str) -> Any:
        """Create a hosted store and return its OpenAI SDK object."""
        # NOTE: OpenAI allows several stores to share one name, so creating
        # a store does not replace an existing one with the same name.
        # Callers that want "only one store called X" must look for an
        # existing store first.
        return await self._run(
            f"create vector store {vector_store_name!r}",
            lambda: self.openai_client.vector_stores.create(name=vector_store_name),
        )

    async def upload_files_and_wait(self, vector_store_id: str, file_paths: list[Path]) -> Any:
        """Upload several files as one batch and wait for indexing, logging progress."""
        # OpenAI's own file_batches.upload_and_poll() does this same thing but
        # silently, with no visibility into progress until it is all done. This
        # unrolls that same sequence - upload concurrently, then poll the batch -
        # so each step can be logged as it happens.

        async def upload_one(file_path: Path, semaphore: asyncio.Semaphore) -> Any:
            async with semaphore:
                # "with ... as ...:" is a context manager: it opens the file,
                # guarantees it gets closed automatically afterward (even if an
                # error happens), and "rb" means "read as raw bytes," which is
                # what the upload API expects. Opening happens inside _run so
                # an unreadable file becomes an adapter error too, rather than
                # a bare OSError escaping to the caller.
                with file_path.open("rb") as file_content:
                    return await self.openai_client.files.create(
                        file=file_content, purpose="assistants"
                    )

        async def upload_and_index() -> Any:
            # Step 1: upload every file concurrently, capped at 5 at once -
            # the same concurrency OpenAI's own upload_and_poll() uses by default.
            semaphore = asyncio.Semaphore(5)
            uploaded_files = await asyncio.gather(
                *(upload_one(file_path, semaphore) for file_path in file_paths)
            )
            logger.info(
                "Uploaded files, starting vector store indexing",
                vector_store_id=vector_store_id,
                file_count=len(uploaded_files),
            )

            # Step 2: attach the uploaded files to the store as one batch, then
            # poll it ourselves instead of calling the SDK's own silent poll,
            # so progress can be logged on every tick.
            file_batch = await self.openai_client.vector_stores.file_batches.create(
                vector_store_id=vector_store_id,
                file_ids=[uploaded_file.id for uploaded_file in uploaded_files],
            )
            while file_batch.file_counts.in_progress > 0:
                await asyncio.sleep(1)
                file_batch = await self.openai_client.vector_stores.file_batches.retrieve(
                    file_batch.id, vector_store_id=vector_store_id
                )
                logger.info(
                    "Vector store indexing progress",
                    vector_store_id=vector_store_id,
                    completed=file_batch.file_counts.completed,
                    failed=file_batch.file_counts.failed,
                    total=file_batch.file_counts.total,
                )
            return file_batch

        return await self._run(
            f"upload {len(file_paths)} files to vector store {vector_store_id}", upload_and_index
        )

    async def delete_vector_store(self, vector_store_id: str) -> bool:
        """Delete a hosted store, returning whether OpenAI confirmed removal."""
        # Deleting is permanent and affects an account-wide resource, so
        # this method only ever acts on the single ID it is given. Nothing
        # in this adapter deletes anything on its own.

        async def delete() -> bool:
            deletion = await self.openai_client.vector_stores.delete(vector_store_id)
            return bool(getattr(deletion, "deleted", False))
            # "getattr(obj, name, default)" reads an attribute only if it
            # exists, so an unexpected response shape reports "not deleted"
            # instead of raising an AttributeError.

        return await self._run(f"delete vector store {vector_store_id}", delete)

    async def close(self) -> None:
        if self._owns_openai_client:
            # Only close the client if THIS object created it - closing a
            # client the caller is still using elsewhere would break their
            # code out from under them.
            await self.openai_client.close()
