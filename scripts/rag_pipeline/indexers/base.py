from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Iterable, Sequence

from ..models import Chunk, ChunkBatch, EvaluationContext, RetrievalResult


class BaseSemanticIndex(ABC):
    """Embedding-powered vector index."""

    @abstractmethod
    def upsert(self, chunks: Sequence[Chunk]) -> None:
        ...

    @abstractmethod
    def delete(self, *, chunk_ids: Sequence[str] | None = None, filter_metadata: dict | None = None) -> None:
        ...

    @abstractmethod
    def query(self, query: str, *, top_k: int = 5, filter_metadata: dict | None = None) -> Sequence[RetrievalResult]:
        ...


class BaseLexicalIndex(ABC):
    """Keyword-based retrieval companion."""

    @abstractmethod
    def upsert(self, chunks: Sequence[Chunk]) -> None:
        ...

    @abstractmethod
    def delete(self, *, chunk_ids: Sequence[str] | None = None, filter_metadata: dict | None = None) -> None:
        ...

    @abstractmethod
    def query(self, query: str, *, top_k: int = 5, filter_metadata: dict | None = None) -> Sequence[RetrievalResult]:
        ...


class HybridIndexer:
    """Coordinate semantic, lexical, and metadata operations."""

    def __init__(
        self,
        semantic_index: BaseSemanticIndex,
        lexical_index: BaseLexicalIndex | None = None,
        evaluation_hook: Callable[[EvaluationContext], EvaluationContext] | None = None,
    ) -> None:
        self.semantic_index = semantic_index
        self.lexical_index = lexical_index
        self._evaluation_hook = evaluation_hook

    def upsert(self, batch: ChunkBatch) -> None:
        self.semantic_index.upsert(batch.chunks)
        if self.lexical_index is not None:
            self.lexical_index.upsert(batch.chunks)

    def delete(
        self,
        *,
        chunk_ids: Sequence[str] | None = None,
        filter_metadata: dict | None = None,
    ) -> None:
        self.semantic_index.delete(chunk_ids=chunk_ids, filter_metadata=filter_metadata)
        if self.lexical_index is not None:
            self.lexical_index.delete(chunk_ids=chunk_ids, filter_metadata=filter_metadata)

    def query_semantic(
        self, query: str, *, top_k: int = 5, filter_metadata: dict | None = None
    ) -> Sequence[RetrievalResult]:
        return self.semantic_index.query(query, top_k=top_k, filter_metadata=filter_metadata)

    def query_hybrid(
        self, query: str, *, top_k: int = 5, filter_metadata: dict | None = None
    ) -> Sequence[RetrievalResult]:
        semantic_results = list(
            self.semantic_index.query(query, top_k=top_k, filter_metadata=filter_metadata)
        )
        if self.lexical_index is None:
            return semantic_results

        lexical_results = list(
            self.lexical_index.query(query, top_k=top_k, filter_metadata=filter_metadata)
        )

        # simple rank fusion: combine unique chunk_ids with max score.
        fused: dict[str, RetrievalResult] = {}
        for result in semantic_results + lexical_results:
            existing = fused.get(result.chunk_id)
            if existing is None or result.score > existing.score:
                fused[result.chunk_id] = result
        return sorted(fused.values(), key=lambda item: item.score, reverse=True)[:top_k]

    def evaluate(self, context: EvaluationContext | None = None) -> EvaluationContext:
        """Dispatch evaluation results to optional callbacks."""
        context = context or EvaluationContext()
        if self._evaluation_hook is not None:
            try:
                return self._evaluation_hook(context)
            except Exception:  # pragma: no cover - evaluation hook should not break ingestion
                return context
        return context


def iter_chunk_batches(chunks: Iterable[Chunk], batch_size: int = 64) -> Iterable[ChunkBatch]:
    buffer: list[Chunk] = []
    for chunk in chunks:
        buffer.append(chunk)
        if len(buffer) >= batch_size:
            yield ChunkBatch(tuple(buffer))
            buffer = []
    if buffer:
        yield ChunkBatch(tuple(buffer))
