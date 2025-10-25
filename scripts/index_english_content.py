"""Composable ingestion pipeline for indexing English MDX content."""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Iterable, List

from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone

from rag_pipeline.chunkers import MarkdownChunker
from rag_pipeline.collectors import MDXCollector
from rag_pipeline.indexers.base import HybridIndexer
from rag_pipeline.indexers.lexical import InMemoryLexicalIndex
from rag_pipeline.indexers.pinecone_semantic import PineconeSemanticIndex
from rag_pipeline.models import EvaluationContext, RawDocument
from rag_pipeline.orchestrator import IngestionOrchestrator, PipelineConfig

ROOT = Path(__file__).resolve().parents[1]
PAGES_DIR = ROOT / "pages"
DEFAULT_NAMESPACE = "prompt-engineering-guide"
DEFAULT_EMBED_MODEL = "text-embedding-3-small"
DEFAULT_REGION = "us-east-1"
MODEL_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}

LOGGER = logging.getLogger("rag_pipeline")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG ingestion pipeline controller")
    subparsers = parser.add_subparsers(dest="command", required=False)

    parser.add_argument("--dry-run", action="store_true", help="Skip index writes")
    parser.add_argument(
        "--evaluation-queries",
        nargs="*",
        help="Optional set of evaluation queries to run post-indexing",
    )

    subparsers.add_parser("full-refresh", help="Rebuild the entire index")

    delete_doc = subparsers.add_parser("delete-doc", help="Remove a single MDX document from the index")
    delete_doc.add_argument("path", help="Relative MDX path, e.g. introduction/index.en.mdx")

    delete_topic = subparsers.add_parser("delete-topic", help="Remove a topic (folder) from the index")
    delete_topic.add_argument("topic", help="Top-level folder under pages/, e.g. techniques")

    upsert_doc = subparsers.add_parser("upsert-doc", help="Upsert a subset of MDX documents")
    upsert_doc.add_argument("paths", nargs="+", help="One or more relative MDX paths")

    return parser.parse_args()


def ensure_env(var_name: str) -> str:
    value = os.environ.get(var_name)
    if not value:
        raise RuntimeError(f"{var_name} is required")
    return value


def build_embeddings(model_name: str, api_key: str) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=model_name, api_key=api_key)


def evaluation_callback(context: EvaluationContext) -> EvaluationContext:
    if context.retriever_results:
        LOGGER.info("Evaluation hook received %d retriever results", len(context.retriever_results))
    return context


def collect_subset(collector: MDXCollector, *, predicate) -> List[RawDocument]:
    documents: List[RawDocument] = []
    for doc in collector.collect():
        if predicate(doc):
            documents.append(doc)
    return documents


def resolve_chunk_ids(documents: Iterable[RawDocument], chunker: MarkdownChunker) -> List[str]:
    chunk_ids: List[str] = []
    for document in documents:
        for chunk in chunker.split(document):
            chunk_ids.append(chunk.chunk_id)
    return chunk_ids


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    args = parse_args()

    openai_api_key = ensure_env("OPENAI_API_KEY")
    pinecone_api_key = ensure_env("PINECONE_API_KEY")
    index_name = ensure_env("PINECONE_INDEX")

    namespace = os.environ.get("PINECONE_NAMESPACE", DEFAULT_NAMESPACE)
    region = os.environ.get("PINECONE_REGION", DEFAULT_REGION)
    embedding_model = os.environ.get("OPENAI_EMBEDDING_MODEL", DEFAULT_EMBED_MODEL)

    embeddings = build_embeddings(embedding_model, openai_api_key)
    dimension = MODEL_DIMENSIONS.get(embedding_model)
    if dimension is None:
        raise RuntimeError(
            f"Unknown embedding dimension for model '{embedding_model}'. Update MODEL_DIMENSIONS to continue."
        )

    pinecone_client = Pinecone(api_key=pinecone_api_key)
    semantic_index = PineconeSemanticIndex(
        pinecone_client=pinecone_client,
        index_name=index_name,
        namespace=namespace,
        embeddings=embeddings,
    )
    semantic_index.ensure_index(dimension=dimension, region=region)

    lexical_index = InMemoryLexicalIndex()
    indexer = HybridIndexer(
        semantic_index=semantic_index,
        lexical_index=lexical_index,
        evaluation_hook=evaluation_callback,
    )

    collector = MDXCollector(PAGES_DIR)
    chunker = MarkdownChunker()
    config = PipelineConfig(
        batch_size=64,
        dry_run=args.dry_run,
        enable_evaluation=bool(args.evaluation_queries),
        evaluation_queries=args.evaluation_queries or [],
    )
    orchestrator = IngestionOrchestrator(
        collector=collector,
        chunker=chunker,
        indexer=indexer,
        config=config,
    )

    command = args.command or "full-refresh"
    if command == "full-refresh":
        orchestrator.run_full_refresh()
    elif command == "delete-doc":
        target_path = args.path
        documents = collect_subset(collector, predicate=lambda doc: doc.metadata["path"] == target_path)
        if not documents:
            LOGGER.warning("No documents found for path '%s'", target_path)
            return
        chunk_ids = resolve_chunk_ids(documents, chunker)
        orchestrator.delete_by_chunk_ids(chunk_ids)
        LOGGER.info("Deleted %d chunks for document %s", len(chunk_ids), target_path)
    elif command == "delete-topic":
        topic = args.topic.rstrip("/")
        documents = collect_subset(
            collector,
            predicate=lambda doc: doc.metadata["path"].startswith(f"{topic}/") or doc.metadata["path"].startswith(topic),
        )
        if not documents:
            LOGGER.warning("No documents found for topic '%s'", topic)
            return
        chunk_ids = resolve_chunk_ids(documents, chunker)
        orchestrator.delete_by_chunk_ids(chunk_ids)
        LOGGER.info("Deleted %d chunks under topic %s", len(chunk_ids), topic)
    elif command == "upsert-doc":
        selected_paths = set(args.paths)
        documents = collect_subset(
            collector,
            predicate=lambda doc: doc.metadata["path"] in selected_paths,
        )
        if not documents:
            LOGGER.warning("No documents matched the provided paths")
            return
        orchestrator.upsert_documents(documents)
        LOGGER.info("Upserted %d documents", len(documents))
    else:
        raise ValueError(f"Unknown command '{command}'")


if __name__ == "__main__":
    main()
