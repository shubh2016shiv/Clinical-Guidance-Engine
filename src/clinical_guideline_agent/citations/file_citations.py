"""Extract file citations from an OpenAI Responses API result.

When the clinical guideline agent answers from uploaded guideline files,
OpenAI returns the answer text PLUS a set of "annotations" saying which
file backed which part of the text. This file digs those annotations out
of OpenAI's reply and returns them as this app's own ``FileCitation``
objects, ready for inline_citations.py to turn into numbered markers
like ``${}^{[1]}$``.

Big-picture flow:
1. Walk the reply's ``output`` list, keeping only the "message" items.
2. Inside each message, keep only the "output_text" content pieces.
3. Inside each piece, keep only the "file_citation" annotations.
4. Skip any citation already recorded for the same file at the same spot.
5. Return the citations in the order they appear in the answer.

Beginner note: the reply is nested like a document - the whole output
holds items, an item holds content pieces, and a piece holds annotations.
Each loop below steps down one level, which is why the code ends up
indented so deeply.
"""

# "Any" is a type hint meaning "this could be any type". It is used here
# because the value comes straight from the OpenAI SDK, whose reply objects
# differ in shape depending on what the model did.
# The ".." means "go up one folder" - FileCitation lives in this component's
# shared schema.py. It records one citation: which file, its name, and the
# character offset in the answer text where the citation belongs.
from typing import Any

from ..schema import FileCitation


# extract_responses_api_file_citations: pulls the citation list out of
# OpenAI's reply. Needed because the reply buries citations three levels
# deep and mixes them with other item types, while the rest of the app just
# wants a flat, ordered list of "which guideline files were cited where".
def extract_responses_api_file_citations(responses_api_result: Any) -> list[FileCitation]:
    """Collect file citations from output text in the order they appear."""
    collected_file_citations: list[FileCitation] = []
    # collected_file_citations: the citations found so far. A list is used
    # (rather than a set) because the ORDER matters - later code numbers the
    # sources by the order they first appear in the answer.
    seen_citation_locations: set[tuple[str, int | None]] = set()
    # seen_citation_locations: the (file_id, offset) pairs already recorded,
    # used to skip exact duplicates. A set makes "have I seen this?" fast.
    # Each member is a tuple because tuples cannot be changed after being
    # created, which is what allows them to be stored in a set at all - a
    # list could not be.
    # Step 1: walk the top-level output items. "getattr(obj, name, default)"
    # reads an attribute only if it exists, returning the default otherwise,
    # so a reply carrying no output does not crash here. The extra "or []"
    # covers the other case: an attribute that EXISTS but is None, which
    # would otherwise make the "for" loop fail.
    for output_item in getattr(responses_api_result, "output", []) or []:
        if getattr(output_item, "type", None) != "message":
            # The output can also hold reasoning summaries and tool calls,
            # which carry no citations. "continue" skips straight to the
            # next item, which keeps the real work below less indented than
            # wrapping all of it in another "if" would.
            continue
        # Step 2: walk the content pieces inside this one message.
        for output_text_content in getattr(output_item, "content", []) or []:
            if getattr(output_text_content, "type", None) != "output_text":
                # Only generated text carries citation annotations.
                continue
            # Step 3: walk the annotations attached to this piece of text.
            for citation_annotation in getattr(output_text_content, "annotations", []) or []:
                if getattr(citation_annotation, "type", None) != "file_citation":
                    # Annotations can also mark other things, such as URL
                    # citations, which this function is not collecting.
                    continue
                file_id = getattr(citation_annotation, "file_id", None)
                # file_id: which uploaded guideline file this citation points
                # at. Without it the citation cannot be tied back to a
                # source, so there is nothing useful to record.
                if not file_id:
                    continue
                annotation_index = getattr(citation_annotation, "index", None)
                # annotation_index: the character offset in the answer text
                # where this citation belongs. inline_citations.py uses it to
                # place the marker. It can be missing, in which case the
                # source is still listed but cannot be positioned.
                citation_location = (file_id, annotation_index)
                # citation_location: "this file, cited at this spot" - the
                # identity used for the duplicate check below.
                # Step 4: skip a citation already recorded for the same file
                # at the same offset, since repeating it would place the same
                # marker twice. The offset is part of the key on purpose: one
                # file cited at two DIFFERENT places is kept twice, because
                # each place gets its own marker.
                if citation_location in seen_citation_locations:
                    continue
                seen_citation_locations.add(citation_location)
                # Step 5: record the citation in this app's own shape, so
                # callers never have to read OpenAI's annotation objects.
                # "filename" falls back to an empty string because it is only
                # used for display.
                collected_file_citations.append(
                    FileCitation(
                        file_id=file_id,
                        filename=getattr(citation_annotation, "filename", ""),
                        index=annotation_index,
                    )
                )
    # Step 6: hand back the citations in the order they were found, which is
    # the order they appear in the answer text.
    return collected_file_citations
