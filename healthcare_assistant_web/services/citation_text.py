"""Makes the orchestrator's citation markers readable on a web page."""

import re

# The orchestrator writes each marker as a LaTeX snippet, e.g. ${}^{[1]}$.
# A web page shows that as raw text, so it is swapped for a raised <sup>[1]</sup>.
_CITATION_MARKER = re.compile(r"\$\{\}\^\{\[(\d+)\]\}\$")


def citation_markers_as_superscript(answer_text: str) -> str:
    """Return the text with every citation marker written as a superscript."""
    return _CITATION_MARKER.sub(r"<sup>[\1]</sup>", answer_text)
