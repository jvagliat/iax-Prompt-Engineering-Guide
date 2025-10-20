# Full-Site RAG Indexing Blueprints

The following scenarios outline opinionated ways to orchestrate an end-to-end ingestion
of the Prompt Engineering Guide web content (English locale) into a vector database. Each
idea highlights the orchestration flow, key components, and implementation notes.

| # | Idea | Orchestration Flow | Key Components |
|---|------|--------------------|----------------|
| 1 | **LangChain + Next.js File Walker** | Traverse the `pages/**\/*.en.mdx` tree with Node.js, normalize MDX to markdown via `remark`, stream content into a Python worker that chunks with `MarkdownTextSplitter`, and push batched embeddings into Pinecone using async tasks. | Node.js file walker, `remark`, LangChain `MarkdownTextSplitter`, OpenAI `text-embedding-3-large`, Pinecone SDK, AsyncIO queue |
| 2 | **LangChain Expression Language (LCEL) Pipeline** | Mount the repo in a Python service, compose a LCEL `RunnableSequence` that loads MDX via `DirectoryLoader`, strips JSX, enriches documents with breadcrumb metadata, chunks, embeds, and writes to Chroma in a single declarative graph. | LangChain DirectoryLoader, custom MDX cleaner, LCEL `RunnableSequence`, `OpenAIEmbeddings`, Chroma |
| 3 | **Airflow-Driven Batch Indexer** | Configure an Airflow DAG with tasks for git clone, MDX parsing, chunking, embedding, and vector upsert; schedule nightly runs and log metrics (chunk counts, token usage) to monitor drift between locales. | Airflow DAG, Dockerized Python operators, `unstructured` MDX partitioner, LangChain `RecursiveCharacterTextSplitter`, OpenAI embeddings, Weaviate |
| 4 | **Dagster Asset Graph** | Model each section (`introduction`, `techniques`, etc.) as Dagster assets; asset materializations run MDX parsing, apply prompt-aware chunking (e.g., by headings), and update Milvus collections with lineage metadata for reproducibility. | Dagster, `mdformat` + AST parsing, LangChain, Milvus (pymilvus), OpenAI embeddings |
| 5 | **dbt + Python Model Hybrid** | Use dbt to orchestrate metadata tables (file paths, slugs, version hashes) and invoke Python models that load MDX, chunk via `MarkdownHeaderTextSplitter`, and write embeddings + metadata into a warehouse-backed pgvector extension. | dbt Core, dbt Python models, LangChain header splitter, OpenAI embeddings, PostgreSQL + pgvector |
| 6 | **Serverless Event-Driven Indexer** | Deploy an AWS Lambda triggered by CodeCommit (or GitHub webhook) that downloads modified `.en.mdx`, runs chunking via `langchain_text_splitters`, embeds with `bedrock` models, and upserts to Amazon OpenSearch Serverless vectors. | AWS Lambda, EventBridge trigger, LangChain text splitters, Amazon Bedrock embeddings, OpenSearch vector index |
| 7 | **Rust Ingestion CLI + Python Embedding Microservice** | Build a Rust CLI using `ignore` crate to efficiently stream MDX files, convert to cleaned markdown, and POST payloads to a FastAPI service that handles chunking, embedding, and Qdrant inserts; supports high throughput and typed schemas. | Rust (`ignore`, `pulldown-cmark`), FastAPI, LangChain `MarkdownTextSplitter`, OpenAI embeddings, Qdrant |
| 8 | **Kafka-Based Streaming Pipeline** | Publish MDX documents as Kafka messages when git changes are detected; a stream processor (Flink or Faust) chunks and enriches, emitting embedding jobs consumed by a vector-writer microservice targeting Redis VL. | Git diff watcher, Kafka, Faust/Flink processor, LangChain splitters, OpenAI embeddings, Redis Vector Similarity |
| 9 | **Prefect Flow with Retriever Evaluation** | Author a Prefect deployment that orchestrates extraction, chunking, and embedding, followed by an automatic evaluation step (e.g., LangChain `QAEvalChain`) to validate retriever quality before committing updates to the vector store. | Prefect 2.0, LangChain, OpenAI embeddings, `QAEvalChain`, LlamaIndex RetrieverEvaluator, Chroma |
| 10 | **LlamaIndex Composable Graph** | Create LlamaIndex `Document` nodes directly from the English MDX files, apply metadata hierarchies via `HierarchicalNodeParser`, and persist to a `VectorStoreIndex` backed by Azure Cognitive Search with scheduled refresh notebooks. | LlamaIndex, `SimpleDirectoryReader` with MDX filter, `HierarchicalNodeParser`, OpenAI embeddings, Azure Cognitive Search |

## Implementation Notes
- **Locale Filtering**: constrain loaders to `*.en.mdx` to avoid indexing non-English content.
- **Metadata Fidelity**: use `_meta.en.json` files to attach section ordering, titles, and navigation breadcrumbs to each chunk.
- **Change Detection**: hash file contents (e.g., SHA256) to enable incremental updates and avoid re-embedding unchanged pages.
- **Evaluation Loop**: pair the vector DB with curated eval questions to monitor retrieval drift after each ingestion run.
