"""Ingestion pipeline: load → preprocess → chunk → index."""

from agentic_rag.ingestion.loaders import (
    BaseLoader,
    DirectoryLoader,
    MarkdownLoader,
    PDFLoader,
    TextLoader,
    WebLoader,
)
from agentic_rag.ingestion.pipeline import IngestionPipeline
from agentic_rag.ingestion.preprocessor import Preprocessor

__all__ = [
    "BaseLoader",
    "TextLoader",
    "MarkdownLoader",
    "PDFLoader",
    "WebLoader",
    "DirectoryLoader",
    "IngestionPipeline",
    "Preprocessor",
]
