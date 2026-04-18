"""Composable ingestion pipeline.

Connects loader → preprocessor → chunker → retriever.index() with a
single entrypoint. Each piece is injectable so it's trivial to unit-test
and to swap implementations.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agentic_rag.core.logging import get_logger
from agentic_rag.ingestion.loaders import BaseLoader
from agentic_rag.ingestion.preprocessor import Preprocessor
from agentic_rag.models.documents import Chunk
from agentic_rag.retrieval.base import BaseRetriever
from agentic_rag.retrieval.chunking import BaseChunker, TokenAwareChunker

log = get_logger(__name__)


@dataclass(slots=True)
class IngestionReport:
    """Summary of a single ingestion run."""

    documents_loaded: int = 0
    chunks_indexed: int = 0
    errors: list[str] = field(default_factory=list)


class IngestionPipeline:
    """End-to-end ingestion executor."""

    def __init__(
        self,
        retriever: BaseRetriever,
        *,
        chunker: BaseChunker | None = None,
        preprocessor: Preprocessor | None = None,
        batch_size: int = 64,
    ) -> None:
        self.retriever = retriever
        self.chunker = chunker or TokenAwareChunker()
        self.preprocessor = preprocessor or Preprocessor()
        self.batch_size = batch_size

    async def ingest(self, loader: BaseLoader) -> IngestionReport:
        report = IngestionReport()
        buffer: list[Chunk] = []

        async for doc in loader.load():
            try:
                doc = self.preprocessor(doc)
                if not doc.content.strip():
                    report.errors.append(f"empty content: {doc.source}")
                    continue
                chunks = self.chunker.split(doc)
                report.documents_loaded += 1
                buffer.extend(chunks)

                if len(buffer) >= self.batch_size:
                    await self.retriever.add(buffer)
                    report.chunks_indexed += len(buffer)
                    buffer.clear()
            except Exception as exc:  # noqa: BLE001
                report.errors.append(f"{doc.source}: {exc}")
                log.bind(source=doc.source).exception("ingestion failed for doc")

        if buffer:
            await self.retriever.add(buffer)
            report.chunks_indexed += len(buffer)
        log.bind(
            docs=report.documents_loaded, chunks=report.chunks_indexed, errors=len(report.errors)
        ).info("ingestion complete")
        return report


__all__ = ["IngestionPipeline", "IngestionReport"]
