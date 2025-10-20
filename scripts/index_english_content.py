"""Quick script to index the English MDX content into Pinecone.

Usage::
    OPENAI_API_KEY=... PINECONE_API_KEY=... PINECONE_INDEX=... python scripts/index_english_content.py

Optional environment variables:
- PINECONE_REGION: Region for index creation (defaults to "us-east-1" if a new index is created).
- PINECONE_NAMESPACE: Namespace to upsert the embeddings (defaults to "prompt-engineering-guide").
- OPENAI_EMBEDDING_MODEL: Embedding model name (defaults to "text-embedding-3-small").

The script walks the `pages/` directory, extracts every `*.en.mdx` file, generates
Markdown-aware chunks, and uploads them to Pinecone together with helpful metadata
(breadcrumbs, slug and locale).
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from langchain_text_splitters import MarkdownTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

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


@dataclass
class ContentChunk:
    document: Document
    id: str


def slug_to_title(slug: str) -> str:
    """Convert slugs such as `prompt-patterns` to title case."""
    return slug.replace("-", " ").replace("_", " ").title()


def load_meta_titles(root: Path) -> Dict[str, str]:
    """Create a lookup {slug_path: title} from every `_meta.en.json` file."""
    lookup: Dict[str, str] = {}

    for meta_path in root.rglob("_meta.en.json"):
        rel_dir = meta_path.parent.relative_to(root)
        with meta_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)

        for slug, entry in data.items():
            if isinstance(entry, str):
                title = entry
            else:
                title = entry.get("title") or entry.get("name") or slug_to_title(slug)

            if str(rel_dir) == ".":
                key = slug
            elif not str(rel_dir):
                key = slug
            else:
                key = f"{rel_dir.as_posix()}/{slug}"
            lookup[key] = title

    return lookup


def extract_breadcrumbs(rel_path: Path, title_lookup: Dict[str, str]) -> Tuple[List[str], str]:
    parts = list(rel_path.parts)
    file_slug = parts[-1].replace(".en.mdx", "")
    slug_parts: List[str] = []
    breadcrumbs: List[str] = []

    for part in parts[:-1]:
        slug_parts.append(part)
        key = "/".join(slug_parts)
        breadcrumbs.append(title_lookup.get(key, slug_to_title(part)))

    slug_parts.append(file_slug)
    key = "/".join(slug_parts)
    title = title_lookup.get(key, slug_to_title(file_slug))
    breadcrumbs.append(title)

    return breadcrumbs, title


def clean_mdx(content: str) -> str:
    """Remove front matter, imports and JSX-only blocks."""
    # Remove front matter.
    if content.startswith("---"):
        content = re.sub(r"^---[\s\S]*?---\s*", "", content, flags=re.MULTILINE)

    # Drop import/export statements that add noise to embeddings.
    content = re.sub(r"^(import|export) .*$", "", content, flags=re.MULTILINE)

    # Remove simple MDX components such as <Component />.
    content = re.sub(r"^<[^>]+>$", "", content, flags=re.MULTILINE)

    return content.strip()


def iter_english_mdx(root: Path) -> Iterable[Path]:
    yield from root.rglob("*.en.mdx")


def build_documents() -> List[ContentChunk]:
    splitter = MarkdownTextSplitter(chunk_size=1200, chunk_overlap=150)
    title_lookup = load_meta_titles(PAGES_DIR)

    chunks: List[ContentChunk] = []
    for file_path in iter_english_mdx(PAGES_DIR):
        rel_path = file_path.relative_to(PAGES_DIR)
        breadcrumbs, title = extract_breadcrumbs(rel_path, title_lookup)

        with file_path.open("r", encoding="utf-8") as fh:
            raw_content = fh.read()

        cleaned = clean_mdx(raw_content)
        if not cleaned:
            continue

        base_metadata = {
            "source": f"/en/{rel_path.as_posix().replace('.en.mdx', '')}",
            "path": rel_path.as_posix(),
            "title": title,
            "breadcrumbs": breadcrumbs,
            "locale": "en",
        }

        documents = splitter.create_documents([cleaned], metadatas=[base_metadata])
        for idx, document in enumerate(documents):
            doc_id = f"{rel_path.as_posix()}::chunk-{idx}"
            chunks.append(ContentChunk(document=document, id=doc_id))

    return chunks


def ensure_index(client: Pinecone, index_name: str, dimension: int, region: str) -> None:
    indexes = {item["name"] for item in client.list_indexes()}
    if index_name in indexes:
        return

    print(f"Creating Pinecone index '{index_name}' with dimension {dimension} in region {region}...")
    client.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",
        spec={
            "serverless": {"cloud": "aws", "region": region},
        },
    )



def main() -> None:
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    index_name = os.environ.get("PINECONE_INDEX")

    if not openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required")
    if not pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY is required")
    if not index_name:
        raise RuntimeError("PINECONE_INDEX is required")

    namespace = os.environ.get("PINECONE_NAMESPACE", DEFAULT_NAMESPACE)
    region = os.environ.get("PINECONE_REGION", DEFAULT_REGION)
    embedding_model = os.environ.get("OPENAI_EMBEDDING_MODEL", DEFAULT_EMBED_MODEL)

    embeddings = OpenAIEmbeddings(model=embedding_model, api_key=openai_api_key)

    dimension = MODEL_DIMENSIONS.get(embedding_model)
    if dimension is None:
        raise RuntimeError(
            f"Unknown embedding dimension for model '{embedding_model}'. Update MODEL_DIMENSIONS to continue."
        )

    client = Pinecone(api_key=pinecone_api_key)
    ensure_index(client, index_name=index_name, dimension=dimension, region=region)

    vector_store = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings,
        namespace=namespace,
    )

    chunks = build_documents()
    if not chunks:
        print("No English MDX content found; aborting.")
        return

    print(f"Uploading {len(chunks)} chunks to Pinecone index '{index_name}' (namespace '{namespace}')...")
    vector_store.add_documents(
        documents=[chunk.document for chunk in chunks],
        ids=[chunk.id for chunk in chunks],
    )
    print("Done!")


if __name__ == "__main__":
    main()
