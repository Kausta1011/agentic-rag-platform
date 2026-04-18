"""Simple text preprocessing: whitespace collapse, boilerplate stripping.

Kept tiny on purpose — aggressive cleaning often hurts retrieval quality
more than it helps (it removes context the embedder could use).
"""

from __future__ import annotations

import re

from agentic_rag.models.documents import Document

_MULTI_WS_RE = re.compile(r"[ \t]+")
_MULTI_NL_RE = re.compile(r"\n{3,}")
_NBSP_RE = re.compile(r"[\u00a0\u2000-\u200b\ufeff]")


class Preprocessor:
    """Normalises whitespace and control characters on a :class:`Document`."""

    def __init__(self, *, collapse_whitespace: bool = True, strip_nbsp: bool = True) -> None:
        self.collapse_whitespace = collapse_whitespace
        self.strip_nbsp = strip_nbsp

    def __call__(self, doc: Document) -> Document:
        text = doc.content
        if self.strip_nbsp:
            text = _NBSP_RE.sub(" ", text)
        if self.collapse_whitespace:
            text = _MULTI_WS_RE.sub(" ", text)
            text = _MULTI_NL_RE.sub("\n\n", text)
        text = text.strip()
        doc.content = text
        return doc


__all__ = ["Preprocessor"]
