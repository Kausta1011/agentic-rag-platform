"""Document loaders.

Adapter pattern — every loader converts an external source (file, URL,
directory) into one or more :class:`Document` instances, hiding the
format-specific detail behind a uniform async interface.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from pathlib import Path

from agentic_rag.core.exceptions import IngestionError
from agentic_rag.core.logging import get_logger
from agentic_rag.models.documents import Document

log = get_logger(__name__)


class BaseLoader(ABC):
    """Loader strategy interface."""

    @abstractmethod
    async def load(self) -> AsyncIterator[Document]:
        """Yield one or more documents.

        An async iterator so loaders streaming large corpora don't need
        to hold everything in memory at once.
        """
        if False:  # pragma: no cover - make this an async generator
            yield  # type: ignore[unreachable]


class TextLoader(BaseLoader):
    """Plain-text / .log / .txt files."""

    def __init__(self, path: str | Path, encoding: str = "utf-8") -> None:
        self.path = Path(path)
        self.encoding = encoding

    async def load(self) -> AsyncIterator[Document]:
        if not self.path.is_file():
            raise IngestionError(f"file not found: {self.path}")
        try:
            content = self.path.read_text(encoding=self.encoding)
        except Exception as exc:  # noqa: BLE001
            raise IngestionError(f"failed reading {self.path}: {exc}") from exc

        yield Document(
            source=str(self.path),
            title=self.path.stem,
            content=content,
            metadata={"loader": "text", "filename": self.path.name},
        )


class MarkdownLoader(TextLoader):
    """Markdown files. Content preserved verbatim — downstream rendering
    is the UI's job; chunking still operates on the raw text."""

    async def load(self) -> AsyncIterator[Document]:
        async for doc in super().load():
            doc.metadata["loader"] = "markdown"
            yield doc


class PDFLoader(BaseLoader):
    """PDF files via pypdf."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    async def load(self) -> AsyncIterator[Document]:
        try:
            from pypdf import PdfReader  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover
            raise IngestionError("pypdf not installed") from exc

        if not self.path.is_file():
            raise IngestionError(f"file not found: {self.path}")

        def _extract() -> str:
            reader = PdfReader(str(self.path))
            return "\n\n".join((page.extract_text() or "").strip() for page in reader.pages)

        try:
            content = await asyncio.to_thread(_extract)
        except Exception as exc:  # noqa: BLE001
            raise IngestionError(f"failed parsing PDF {self.path}: {exc}") from exc

        yield Document(
            source=str(self.path),
            title=self.path.stem,
            content=content,
            metadata={"loader": "pdf", "filename": self.path.name},
        )


class WebLoader(BaseLoader):
    """Fetch an HTML page and extract its visible text."""

    def __init__(self, url: str, timeout: float = 15.0) -> None:
        self.url = url
        self.timeout = timeout

    async def load(self) -> AsyncIterator[Document]:
        try:
            import httpx
            from bs4 import BeautifulSoup
        except ImportError as exc:  # pragma: no cover
            raise IngestionError("httpx / beautifulsoup4 not installed") from exc

        try:
            async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=True) as client:
                resp = await client.get(self.url, headers={"User-Agent": "agentic-rag/0.1"})
                resp.raise_for_status()
                html = resp.text
        except Exception as exc:  # noqa: BLE001
            raise IngestionError(f"failed fetching {self.url}: {exc}") from exc

        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = "\n".join(line.strip() for line in soup.get_text().splitlines() if line.strip())
        title = soup.title.get_text(strip=True) if soup.title else self.url

        yield Document(
            source=self.url,
            title=title,
            content=text,
            metadata={"loader": "web", "url": self.url},
        )


class DirectoryLoader(BaseLoader):
    """Walk a directory and dispatch to per-extension loaders."""

    _EXT_MAP: dict[str, type[BaseLoader]] = {
        ".txt": TextLoader,
        ".log": TextLoader,
        ".md": MarkdownLoader,
        ".markdown": MarkdownLoader,
        ".pdf": PDFLoader,
    }

    def __init__(self, root: str | Path, *, recursive: bool = True) -> None:
        self.root = Path(root)
        self.recursive = recursive

    async def load(self) -> AsyncIterator[Document]:
        if not self.root.is_dir():
            raise IngestionError(f"not a directory: {self.root}")
        pattern = "**/*" if self.recursive else "*"
        for path in sorted(self.root.glob(pattern)):
            if not path.is_file():
                continue
            loader_cls = self._EXT_MAP.get(path.suffix.lower())
            if loader_cls is None:
                log.bind(path=str(path)).debug("skipping unsupported extension")
                continue
            loader = loader_cls(path)
            async for doc in loader.load():
                yield doc


__all__ = [
    "BaseLoader",
    "TextLoader",
    "MarkdownLoader",
    "PDFLoader",
    "WebLoader",
    "DirectoryLoader",
]
