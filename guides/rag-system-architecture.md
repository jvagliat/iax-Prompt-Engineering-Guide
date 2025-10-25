# RAG Ingestion & Retrieval Architecture

## Overview
This document captures the target architecture for refactoring the ingestion and indexing pipeline into modular, testable components. The design supports hybrid retrieval (semantic + lexical), rich metadata filtering, and lifecycle operations (create/update/delete) across the corpus of English MDX guides.

## Goals
- **Separation of concerns**: isolate filesystem traversal, chunk preparation, and indexing so each concern can evolve independently.
- **Hybrid retrieval ready**: treat semantic embeddings, lexical indexes, and metadata as first-class representations.
- **Operational ergonomics**: enable partial reindexing, topic-level deletes, and change-tracked updates without full reloads.
- **Extensibility**: add new corpora, embedding providers, or storage backends with minimal code churn.

## Component Model

```mermaid
flowchart LR
    subgraph DataLayer
        A[MDXCollector\nTraversal & normalization]
        B[MarkdownChunker\nStructure-aware splitter]
    end

    subgraph IndexingLayer
        C[HybridIndexer]
        C1[SemanticEmbedder\n(OpenAI, etc.)]
        C2[LexicalIndexer\n(BM25 / keyword)]
        C3[MetadataStore\n(topic, breadcrumbs, version)]
    end

    subgraph ControlPlane
        D[IngestionOrchestrator\nCLI / scheduler]
        E[ChangeTracker\n(Git diff / FS watch)]
        F[EvaluationRunner\n(Retrieval QA / smoke tests)]
    end

    A --> B --> C
    C --> C1
    C --> C2
    C --> C3
    D --> A
    D --> C
    D --> F
    E --> D
```

### MDXCollector
- Discovers `.en.mdx` files based on configured root paths.
- Normalizes frontmatter, imports, custom components, and extracts canonical identifiers.
- Emits `RawDocument` objects with fields: `doc_id`, `source_path`, `title`, `breadcrumbs`, `raw_markdown`, `tags`.

### MarkdownChunker
- Applies markdown-aware splitting (e.g., `RecursiveCharacterTextSplitter` or `MarkdownHeaderTextSplitter`).
- Preserves heading hierarchy in metadata, merging short paragraphs to maintain context length.
- Produces `Chunk` objects with `chunk_id`, `doc_id`, `text`, `metadata` (heading, breadcrumb, ordering, token_count).

### HybridIndexer
Responsible for persisting documents and chunks across multiple retrieval modalities.

Subcomponents:
- **SemanticEmbedder**: wraps embedding providers; exposes `embed_documents(chunks)` and `embed_query(text)`.
- **LexicalIndexer**: maintains keyword/BM25 structures (e.g., Pinecone pod, OpenSearch). Supports `index(chunks)`, `delete(doc_id/topic)`, and query methods.
- **MetadataStore**: single source of truth for chunk metadata, storing version, timestamps, and custom filters.

Public interface:
- `upsert_documents(raw_docs: List[RawDocument])`
- `delete_by_doc_id(doc_ids: List[str])`
- `delete_by_topic(topic: str)`
- `reindex_documents(doc_ids: List[str])`
- `query_semantic(query: str, filters: Dict)`
- `query_hybrid(query: str, filters: Dict, weights: HybridWeights)`

### IngestionOrchestrator
- Coordinates execution steps: collect → chunk → index → verify.
- Offers CLI commands/jobs for full sync, incremental sync, topic rebuilds, and dry runs.
- Handles configuration (environment variables, chunk sizes, providers) and structured logging.

### ChangeTracker
- Detects modified or deleted MDX sources using Git metadata or filesystem timestamps.
- Emits tasks to the orchestrator for targeted updates or deletions.

### EvaluationRunner
- Executes smoke tests (e.g., retrieval QA sets) after indexing.
- Stores evaluation metrics to track regression or improvement over time.

## Data Flow
1. **Discovery**: Orchestrator invokes `MDXCollector.collect()` to obtain `RawDocument` instances.
2. **Chunking**: Each `RawDocument` is passed to `MarkdownChunker.split()` generating ordered `Chunk`s.
3. **Indexing**: `HybridIndexer.upsert_documents()` stores metadata, pushes embeddings to vector store, and updates lexical index.
4. **Verification**: Orchestrator triggers `EvaluationRunner.run_suite()` to ensure retrieval quality.
5. **Change Management**: ChangeTracker monitors sources; on diffs it calls `delete_by_doc_id` + `upsert_documents` for refresh, or `delete_by_topic` when entire sections are replaced.

## Metadata Strategy
- **Core fields**: `doc_id`, `chunk_id`, `title`, `breadcrumbs`, `topic`, `tags`, `source_path`, `last_updated`.
- **Filters**: boolean (e.g., `is_advanced`), categorical (`section`), temporal (`last_updated`).
- **Lexical hints**: store code blocks and key terms separately to improve keyword matching.

## Operational Modes
- **Full Rebuild**: wipe indexes, re-run complete pipeline.
- **Incremental Update**: detect changed docs, soft-delete old chunks, upsert new ones.
- **Topic Refresh**: remove all documents under a breadcrumb/topic and replace with new material.
- **Evaluation-only**: run retrieval QA without touching indexes (useful after model/config changes).

## Extensibility Hooks
- Plug-in new corpus collectors by implementing the `Collector` interface.
- Support additional embedding providers by registering new `SemanticEmbedder` adapters.
- Swap lexical backends (e.g., Elastic, Typesense) through the `LexicalIndexer` interface.
- Integrate GraphRAG by adding a `GraphBuilder` component parallel to `HybridIndexer` that constructs knowledge graphs from the same `RawDocument` stream.

## Next Steps
1. Implement interfaces and dataclasses for `RawDocument`, `Chunk`, and configuration entities. ✅
2. Refactor existing `index_english_content.py` into modular packages following this design. ✅
3. Add unit tests for collector normalization and chunking heuristics.
4. Provide CLI entry points for orchestrator commands and document operational playbooks. ✅
5. Integrate retrieval/generation evaluation harnesses (e.g., RAGAS) via the `HybridIndexer` evaluation hook.

### Implementation snapshot
- [`scripts/rag_pipeline/models.py`](../scripts/rag_pipeline/models.py) defines the core dataclasses shared across collectors, chunkers, and indexers.
- [`scripts/rag_pipeline/collectors.py`](../scripts/rag_pipeline/collectors.py) exposes `MDXCollector`, isolating filesystem traversal and metadata normalization for English MDX files.
- [`scripts/rag_pipeline/chunkers.py`](../scripts/rag_pipeline/chunkers.py) wraps LangChain's `MarkdownTextSplitter` and injects chunk identifiers into metadata to streamline deletes.
- [`scripts/rag_pipeline/indexers/`](../scripts/rag_pipeline/indexers/) hosts `HybridIndexer`, a Pinecone-backed semantic adapter, and an in-memory lexical index ready to be swapped for Elastic, Qdrant, or Typesense implementations.
- [`scripts/rag_pipeline/orchestrator.py`](../scripts/rag_pipeline/orchestrator.py) coordinates batch ingestion, targeted upserts/deletes, and optional evaluation passes.
- [`scripts/index_english_content.py`](../scripts/index_english_content.py) now provides a CLI for full refreshes, targeted updates, and dry-run evaluation smoke tests.
