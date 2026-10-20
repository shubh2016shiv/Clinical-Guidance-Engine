"""Read file citations from typed Responses output items."""

from typing import Any

from ..schema import FileCitation


def file_citations(response: Any) -> list[FileCitation]:
    """Collect citations from message output text, preserving their order."""
    citations: list[FileCitation] = []
    seen: set[tuple[str, int | None]] = set()
    for item in getattr(response, "output", []) or []:
        if getattr(item, "type", None) != "message":
            continue
        for content in getattr(item, "content", []) or []:
            if getattr(content, "type", None) != "output_text":
                continue
            for annotation in getattr(content, "annotations", []) or []:
                if getattr(annotation, "type", None) != "file_citation":
                    continue
                file_id = getattr(annotation, "file_id", None)
                if not file_id:
                    continue
                index = getattr(annotation, "index", None)
                key = (file_id, index)
                if key in seen:
                    continue
                seen.add(key)
                citations.append(
                    FileCitation(
                        file_id=file_id,
                        filename=getattr(annotation, "filename", ""),
                        index=index,
                    )
                )
    return citations
