from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, List

from langchain_text_splitters import MarkdownTextSplitter

from .models import Chunk, RawDocument


class Chunker(ABC):
    """Interface for splitting raw documents into indexable chunks."""

    @abstractmethod
    def split(self, document: RawDocument) -> Iterable[Chunk]:
        """Yield chunks for the provided document."""


class MarkdownChunker(Chunker):
    """LangChain-based Markdown chunker with configurable parameters."""

    def __init__(self, chunk_size: int = 1200, chunk_overlap: int = 150) -> None:
        self._splitter = MarkdownTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def split(self, document: RawDocument) -> Iterable[Chunk]:
        docs = self._splitter.create_documents([document.content], metadatas=[document.metadata])
        chunks: List[Chunk] = []
        for idx, langchain_doc in enumerate(docs):
            chunk_id = f"{document.doc_id}::chunk-{idx}"
            metadata = dict(langchain_doc.metadata)
            metadata["chunk_id"] = chunk_id
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    text=langchain_doc.page_content,
                    metadata=metadata,
                )
            )
        return chunks
