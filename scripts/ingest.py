"""CLI: ingest documents into the vector store.

Usage::

    python -m scripts.ingest --path data/corpus
    python -m scripts.ingest --url https://example.com/article
    python -m scripts.ingest --file paper.pdf

Keeps the ingestion path trivially operable in production (cron / CI).
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from agentic_rag.core.logging import configure_logging, get_logger
from agentic_rag.ingestion.loaders import (
    BaseLoader,
    DirectoryLoader,
    PDFLoader,
    TextLoader,
    WebLoader,
)
from agentic_rag.ingestion.pipeline import IngestionPipeline
from agentic_rag.llm import get_embedding_provider
from agentic_rag.retrieval.bm25_retriever import BM25Retriever
from agentic_rag.retrieval.hybrid import HybridRetriever
from agentic_rag.retrieval.vector_store import ChromaVectorStore

log = get_logger(__name__)


def _pick_loader(args: argparse.Namespace) -> BaseLoader:
    if args.url:
        return WebLoader(args.url)
    if args.file:
        p = Path(args.file)
        return PDFLoader(p) if p.suffix.lower() == ".pdf" else TextLoader(p)
    if args.path:
        return DirectoryLoader(args.path, recursive=True)
    raise SystemExit("must pass one of --url / --file / --path")


async def _run(args: argparse.Namespace) -> None:
    configure_logging()
    embeddings = get_embedding_provider()
    vector = ChromaVectorStore(embeddings=embeddings)
    bm25 = BM25Retriever()
    retriever = HybridRetriever([vector, bm25])
    pipeline = IngestionPipeline(retriever)

    loader = _pick_loader(args)
    report = await pipeline.ingest(loader)
    print(
        f"\nIngested:\n"
        f"  documents : {report.documents_loaded}\n"
        f"  chunks    : {report.chunks_indexed}\n"
        f"  errors    : {len(report.errors)}"
    )
    for err in report.errors[:10]:
        print(f"    ! {err}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into the RAG corpus")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--path", help="Directory to recursively ingest")
    src.add_argument("--file", help="Single file (txt/md/pdf)")
    src.add_argument("--url", help="Web URL to fetch and ingest")
    args = parser.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":  # pragma: no cover
    main()


__all__ = ["main"]
