from __future__ import annotations

from typing import Sequence

from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

from ..models import Chunk, RetrievalResult
from .base import BaseSemanticIndex


class PineconeSemanticIndex(BaseSemanticIndex):
    """Semantic index backed by Pinecone via LangChain."""

    def __init__(
        self,
        *,
        pinecone_client: Pinecone,
        index_name: str,
        namespace: str,
        embeddings: Embeddings | None = None,
    ) -> None:
        self._client = pinecone_client
        self._index_name = index_name
        self._namespace = namespace
        self._embeddings = embeddings or OpenAIEmbeddings()
        self._vector_store = PineconeVectorStore(
            index_name=index_name,
            embedding=self._embeddings,
            namespace=namespace,
        )

    @property
    def vector_store(self) -> PineconeVectorStore:
        return self._vector_store

    def upsert(self, chunks: Sequence[Chunk]) -> None:
        documents = []
        ids = []
        for chunk in chunks:
            documents.append((chunk.text, chunk.metadata))
            ids.append(chunk.chunk_id)
        if not documents:
            return
        texts, metadatas = zip(*documents)
        self._vector_store.add_texts(list(texts), metadatas=list(metadatas), ids=list(ids))

    def delete(
        self,
        *,
        chunk_ids: Sequence[str] | None = None,
        filter_metadata: dict | None = None,
    ) -> None:
        self._vector_store.delete(ids=None if chunk_ids is None else list(chunk_ids), filter=filter_metadata)

    def query(
        self,
        query: str,
        *,
        top_k: int = 5,
        filter_metadata: dict | None = None,
    ) -> Sequence[RetrievalResult]:
        results = self._vector_store.similarity_search_with_score(query, k=top_k, filter=filter_metadata)
        retrievals: list[RetrievalResult] = []
        for doc, score in results:
            chunk_id = doc.metadata.get("chunk_id") or doc.metadata.get("id") or doc.metadata.get("chunkId")
            if chunk_id is None:
                chunk_id = doc.metadata.get("path", "unknown")
            retrievals.append(
                RetrievalResult(
                    chunk_id=chunk_id,
                    score=float(score),
                    metadata=doc.metadata,
                )
            )
        return retrievals

    def ensure_index(self, *, dimension: int, region: str) -> None:
        indexes = {item["name"] for item in self._client.list_indexes()}
        if self._index_name in indexes:
            return
        self._client.create_index(
            name=self._index_name,
            dimension=dimension,
            metric="cosine",
            spec={"serverless": {"cloud": "aws", "region": region}},
        )

