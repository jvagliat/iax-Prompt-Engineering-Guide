from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Iterable, Iterator, List

from .models import RawDocument


class DocumentCollector(ABC):
    """Interface for fetching raw documents prior to chunking."""

    @abstractmethod
    def collect(self) -> Iterator[RawDocument]:
        """Yield the documents to index."""


class MDXCollector(DocumentCollector):
    """Traverse MDX content and yield English-localised documents."""

    def __init__(self, root: Path, locale: str = "en") -> None:
        self.root = root
        self.locale = locale
        self._title_lookup: Dict[str, str] | None = None

    def collect(self) -> Iterator[RawDocument]:
        for file_path in self._iter_locale_files():
            rel_path = file_path.relative_to(self.root)
            breadcrumbs, title = self._extract_breadcrumbs(rel_path)
            with file_path.open("r", encoding="utf-8") as fh:
                content = fh.read()

            cleaned = self._clean_mdx(content)
            if not cleaned:
                continue

            doc_id = rel_path.as_posix()
            metadata = {
                "source": f"/{self.locale}/{rel_path.as_posix().replace(f'.{self.locale}.mdx', '')}",
                "path": rel_path.as_posix(),
                "title": title,
                "breadcrumbs": breadcrumbs,
                "locale": self.locale,
            }
            yield RawDocument(doc_id=doc_id, path=file_path, content=cleaned, metadata=metadata)

    # Internal helpers -------------------------------------------------
    def _iter_locale_files(self) -> Iterable[Path]:
        pattern = f"*.{self.locale}.mdx"
        return self.root.rglob(pattern)

    def _clean_mdx(self, content: str) -> str:
        if content.startswith("---"):
            content = re.sub(r"^---[\s\S]*?---\s*", "", content, flags=re.MULTILINE)
        content = re.sub(r"^(import|export) .*$", "", content, flags=re.MULTILINE)
        content = re.sub(r"^<[^>]+>$", "", content, flags=re.MULTILINE)
        return content.strip()

    def _extract_breadcrumbs(self, rel_path: Path) -> tuple[List[str], str]:
        title_lookup = self._ensure_title_lookup()
        parts = list(rel_path.parts)
        file_slug = parts[-1].replace(f".{self.locale}.mdx", "")
        slug_parts: List[str] = []
        breadcrumbs: List[str] = []

        for part in parts[:-1]:
            slug_parts.append(part)
            key = "/".join(slug_parts)
            breadcrumbs.append(title_lookup.get(key, self._slug_to_title(part)))

        slug_parts.append(file_slug)
        key = "/".join(slug_parts)
        title = title_lookup.get(key, self._slug_to_title(file_slug))
        breadcrumbs.append(title)
        return breadcrumbs, title

    def _slug_to_title(self, slug: str) -> str:
        return slug.replace("-", " ").replace("_", " ").title()

    def _ensure_title_lookup(self) -> Dict[str, str]:
        if self._title_lookup is not None:
            return self._title_lookup

        lookup: Dict[str, str] = {}
        for meta_path in self.root.rglob(f"_meta.{self.locale}.json"):
            rel_dir = meta_path.parent.relative_to(self.root)
            with meta_path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            for slug, entry in data.items():
                if isinstance(entry, str):
                    title = entry
                else:
                    title = entry.get("title") or entry.get("name") or self._slug_to_title(slug)

                if str(rel_dir) in {".", ""}:
                    key = slug
                else:
                    key = f"{rel_dir.as_posix()}/{slug}"
                lookup[key] = title

        self._title_lookup = lookup
        return lookup
