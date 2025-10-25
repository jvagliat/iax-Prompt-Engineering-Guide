from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from .chunkers import Chunker
from .collectors import DocumentCollector
from .indexers.base import HybridIndexer, iter_chunk_batches
from .models import Chunk, EvaluationContext, RawDocument

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    batch_size: int = 64
    dry_run: bool = False
    enable_evaluation: bool = False
    evaluation_queries: Sequence[str] = field(default_factory=list)


class IngestionOrchestrator:
    """Coordinates document collection, chunking, indexing, and evaluation."""

    def __init__(
        self,
        *,
        collector: DocumentCollector,
        chunker: Chunker,
        indexer: HybridIndexer,
        config: PipelineConfig | None = None,
    ) -> None:
        self.collector = collector
        self.chunker = chunker
        self.indexer = indexer
        self.config = config or PipelineConfig()

    def run_full_refresh(self) -> None:
        logger.info("Starting full refresh pipeline")
        documents = list(self._collect_documents())
        logger.info("Collected %d documents", len(documents))

        all_chunks: list[Chunk] = []
        for document in documents:
            chunks = list(self.chunker.split(document))
            all_chunks.extend(chunks)
        logger.info("Generated %d chunks", len(all_chunks))

        if self.config.dry_run:
            logger.info("Dry-run enabled; skipping indexing step")
            return

        for batch in iter_chunk_batches(all_chunks, batch_size=self.config.batch_size):
            self.indexer.upsert(batch)
        logger.info("Upserted %d chunks", len(all_chunks))

        if self.config.enable_evaluation and self.config.evaluation_queries:
            self._run_evaluations()

    def upsert_documents(self, documents: Iterable[RawDocument]) -> None:
        chunks: list[Chunk] = []
        for document in documents:
            chunks.extend(list(self.chunker.split(document)))
        for batch in iter_chunk_batches(chunks, batch_size=self.config.batch_size):
            self.indexer.upsert(batch)

    def delete_by_chunk_ids(self, chunk_ids: Sequence[str]) -> None:
        self.indexer.delete(chunk_ids=chunk_ids)

    def delete_by_metadata(self, metadata_filter: dict) -> None:
        self.indexer.delete(filter_metadata=metadata_filter)

    def _collect_documents(self) -> Iterable[RawDocument]:
        return list(self.collector.collect())

    def _run_evaluations(self) -> EvaluationContext:
        context = EvaluationContext()
        logger.info("Running evaluation queries: %s", ", ".join(self.config.evaluation_queries))
        results: list = []
        for query in self.config.evaluation_queries:
            results.extend(self.indexer.query_hybrid(query, top_k=5))
        context = self.indexer.evaluate(EvaluationContext(retriever_results=results))
        logger.info("Evaluation complete with %d retrieval results", len(results))
        return context
