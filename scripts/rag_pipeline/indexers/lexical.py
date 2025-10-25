from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Dict, Iterable, Sequence

from ..models import Chunk, RetrievalResult
from .base import BaseLexicalIndex

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")


def _tokenize(text: str) -> Iterable[str]:
    for match in _TOKEN_PATTERN.finditer(text.lower()):
        yield match.group(0)


class InMemoryLexicalIndex(BaseLexicalIndex):
    """Naive BM25-like index to support hybrid retrieval flows."""

    def __init__(self) -> None:
        self._documents: Dict[str, Dict[str, float]] = {}
        self._metadata: Dict[str, dict] = {}
        self._doc_length: Dict[str, int] = {}
        self._term_document_counts: Counter[str] = Counter()

    def upsert(self, chunks: Sequence[Chunk]) -> None:
        for chunk in chunks:
            tokens = list(_tokenize(chunk.text))
            if not tokens:
                continue
            term_freq = Counter(tokens)
            self._remove_chunk(chunk.chunk_id)
            self._documents[chunk.chunk_id] = term_freq
            self._metadata[chunk.chunk_id] = chunk.metadata
            self._doc_length[chunk.chunk_id] = len(tokens)
            self._term_document_counts.update(set(term_freq))

    def delete(
        self,
        *,
        chunk_ids: Sequence[str] | None = None,
        filter_metadata: dict | None = None,
    ) -> None:
        ids = chunk_ids or list(self._documents)
        for chunk_id in ids:
            metadata = self._metadata.get(chunk_id)
            if filter_metadata and metadata:
                if not all(metadata.get(k) == v for k, v in filter_metadata.items()):
                    continue
            self._remove_chunk(chunk_id)

    def query(
        self,
        query: str,
        *,
        top_k: int = 5,
        filter_metadata: dict | None = None,
    ) -> Sequence[RetrievalResult]:
        tokens = list(_tokenize(query))
        if not tokens:
            return []

        scores: Dict[str, float] = defaultdict(float)
        avg_doc_len = (
            sum(self._doc_length.values()) / len(self._doc_length) if self._doc_length else 0.0
        )
        for token in tokens:
            doc_freq = self._term_document_counts[token]
            if doc_freq == 0:
                continue
            idf = math.log(1 + (len(self._documents) - doc_freq + 0.5) / (doc_freq + 0.5))
            for chunk_id, term_freqs in self._documents.items():
                metadata = self._metadata.get(chunk_id)
                if filter_metadata and metadata:
                    if not all(metadata.get(k) == v for k, v in filter_metadata.items()):
                        continue
                freq = term_freqs.get(token, 0)
                if freq == 0:
                    continue
                doc_len = self._doc_length.get(chunk_id, 1)
                score = idf * ((freq * (1.5 + 1)) / (freq + 1.5 * (1 - 0.75 + 0.75 * doc_len / (avg_doc_len or 1))))
                scores[chunk_id] += score

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
        return [
            RetrievalResult(chunk_id=chunk_id, score=score, metadata=self._metadata.get(chunk_id, {}))
            for chunk_id, score in ranked
        ]

    # ------------------------------------------------------------------
    def _remove_chunk(self, chunk_id: str) -> None:
        term_freq = self._documents.pop(chunk_id, None)
        if term_freq is None:
            return
        for token in term_freq:
            self._term_document_counts[token] -= 1
            if self._term_document_counts[token] <= 0:
                del self._term_document_counts[token]
        self._metadata.pop(chunk_id, None)
        self._doc_length.pop(chunk_id, None)

