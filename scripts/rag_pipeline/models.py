from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence


@dataclass(frozen=True)
class RawDocument:
    """Representation of a source document prior to chunking."""

    doc_id: str
    path: Path
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    """Atomic unit that will be indexed."""

    chunk_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ChunkBatch:
    """Utility wrapper for grouped chunk operations."""

    chunks: Sequence[Chunk]

    def __post_init__(self) -> None:
        if not self.chunks:
            raise ValueError("ChunkBatch requires at least one chunk")

    def chunk_ids(self) -> List[str]:
        return [chunk.chunk_id for chunk in self.chunks]

    def documents_metadata(self) -> List[Dict[str, Any]]:
        return [chunk.metadata for chunk in self.chunks]


@dataclass(frozen=True)
class RetrievalResult:
    """Placeholder for retrieval responses, enabling evaluation hooks."""

    chunk_id: str
    score: float
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class EvaluationContext:
    """Container for evaluation artefacts generated during ingestion."""

    retriever_results: Sequence[RetrievalResult] | None = None
    generator_responses: Sequence[str] | None = None
    notes: Dict[str, Any] | None = None
