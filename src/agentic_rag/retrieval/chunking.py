"""Chunking strategies.

Two strategies ship out of the box:

* :class:`RecursiveChunker` — character-based, splits on a hierarchy of
  separators (paragraph → sentence → word). Cheap, no tokeniser.
* :class:`TokenAwareChunker` — token-budget-aware splitter that uses
  tiktoken when available. Preferred for production because it respects
  the context window precisely.

Both produce :class:`Chunk` objects with stable deterministic ids.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from agentic_rag.models.documents import Chunk, Document


class BaseChunker(ABC):
    """Chunker strategy interface."""

    @abstractmethod
    def split(self, document: Document) -> list[Chunk]: ...


class RecursiveChunker(BaseChunker):
    """Recursive character splitter — no tokeniser dependency."""

    _DEFAULT_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", " ", "")

    def __init__(
        self,
        chunk_size: int = 1_000,
        chunk_overlap: int = 150,
        separators: Sequence[str] | None = None,
    ) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be < chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = tuple(separators) if separators else self._DEFAULT_SEPARATORS

    # ------------------------------------------------------------------
    def split(self, document: Document) -> list[Chunk]:
        pieces = self._recursive_split(document.content, list(self.separators))
        merged = self._merge_with_overlap(pieces)
        return [
            Chunk(
                id=Chunk.build_id(document.id, i),
                document_id=document.id,
                ordinal=i,
                content=text,
                metadata={**document.metadata, "source": document.source, "title": document.title},
            )
            for i, text in enumerate(merged)
        ]

    # ------------------------------------------------------------------
    def _recursive_split(self, text: str, separators: list[str]) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size or not separators:
            return [text]

        sep, *rest = separators
        if sep == "":
            # Hard fallback — slice by chunk_size.
            return [text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

        splits = text.split(sep)
        out: list[str] = []
        for s in splits:
            piece = s if sep == "" else s + sep
            if len(piece) <= self.chunk_size:
                out.append(piece)
            else:
                out.extend(self._recursive_split(piece, rest))
        return [p for p in out if p.strip()]

    def _merge_with_overlap(self, pieces: list[str]) -> list[str]:
        """Greedy merge of small pieces into ~chunk_size windows with overlap."""
        chunks: list[str] = []
        buffer = ""
        for p in pieces:
            if len(buffer) + len(p) <= self.chunk_size:
                buffer += p
                continue
            if buffer:
                chunks.append(buffer.strip())
                # Start next buffer with tail overlap from previous buffer.
                tail = buffer[-self.chunk_overlap :] if self.chunk_overlap else ""
                buffer = tail + p
            else:
                buffer = p
        if buffer.strip():
            chunks.append(buffer.strip())
        return chunks


class TokenAwareChunker(BaseChunker):
    """Token-budget chunker using tiktoken when available.

    Falls back to :class:`RecursiveChunker` behaviour if tiktoken is
    unavailable — this keeps dev / CI environments happy.
    """

    def __init__(
        self,
        max_tokens: int = 512,
        overlap_tokens: int = 64,
        encoding_name: str = "cl100k_base",
    ) -> None:
        if overlap_tokens >= max_tokens:
            raise ValueError("overlap_tokens must be < max_tokens")
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

        try:
            import tiktoken  # type: ignore[import-not-found]

            self._enc = tiktoken.get_encoding(encoding_name)
            self._fallback: BaseChunker | None = None
        except Exception:  # pragma: no cover - safety net
            self._enc = None
            self._fallback = RecursiveChunker(
                chunk_size=max_tokens * 4, chunk_overlap=overlap_tokens * 4
            )

    def split(self, document: Document) -> list[Chunk]:
        if self._enc is None:
            assert self._fallback is not None
            return self._fallback.split(document)

        tokens = self._enc.encode(document.content)
        step = self.max_tokens - self.overlap_tokens
        windows: list[list[int]] = []
        for start in range(0, len(tokens), step):
            window = tokens[start : start + self.max_tokens]
            if not window:
                break
            windows.append(window)
            if start + self.max_tokens >= len(tokens):
                break

        return [
            Chunk(
                id=Chunk.build_id(document.id, i),
                document_id=document.id,
                ordinal=i,
                content=self._enc.decode(window),
                metadata={**document.metadata, "source": document.source, "title": document.title},
            )
            for i, window in enumerate(windows)
        ]


__all__ = ["BaseChunker", "RecursiveChunker", "TokenAwareChunker"]
