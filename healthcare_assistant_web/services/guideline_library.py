"""Uploading, listing and deleting the guideline files in the shared store."""

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from src.clinical_guideline_agent import IndexedGuidelineFile

from ..web_ui_settings import GUIDELINE_FILE_SUFFIXES


@dataclass(frozen=True)
class UploadResult:
    """Which uploaded files were indexed, and which were skipped as the wrong type."""

    indexed_names: list[str]
    skipped_names: list[str]


class GuidelineLibrary:
    """The guideline files a user can see and manage, backed by the vector store."""

    def __init__(self, guideline_agent: Any) -> None:
        self.guideline_agent = guideline_agent

    async def list_files(self) -> list[IndexedGuidelineFile]:
        return await self.guideline_agent.list_guideline_files()

    async def upload_files(self, uploads: list[Any]) -> UploadResult:
        """Index the PDF and Markdown files among the uploads.

        Each upload only needs a ``name`` and an async ``save(path)``, which is
        what the NiceGUI upload gives us.
        """
        accepted_uploads = [
            upload
            for upload in uploads
            if Path(upload.name).suffix.lower() in GUIDELINE_FILE_SUFFIXES
        ]
        skipped_names = [upload.name for upload in uploads if upload not in accepted_uploads]
        if not accepted_uploads:
            return UploadResult(indexed_names=[], skipped_names=skipped_names)

        # The files only need to exist on disk while they are sent to OpenAI.
        with TemporaryDirectory() as temporary_directory:
            saved_paths: list[Path] = []
            for upload_number, upload in enumerate(accepted_uploads):
                # A folder per file keeps two files with the same name apart.
                file_path = Path(temporary_directory) / str(upload_number) / Path(upload.name).name
                await upload.save(file_path)
                saved_paths.append(file_path)

            await self.guideline_agent.add_guideline_files(saved_paths)

        return UploadResult(
            indexed_names=[upload.name for upload in accepted_uploads],
            skipped_names=skipped_names,
        )

    async def delete_file(self, file_id: str) -> None:
        await self.guideline_agent.remove_guideline_file(file_id)
