"""Human-readable Markdown persistence and deterministic BM25 recall."""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
import json
import logging
import math
import os
from pathlib import Path
import re
import threading
from uuid import uuid4

from personal_knowledge.retrieval.lexical import tokenize

from .models import MemoryEntry, MemoryKind, MemoryRecall


logger = logging.getLogger(__name__)

_SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"(?i)\b(?:api[_ -]?key|access[_ -]?token|password)\s*[:=]\s*\S+"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~-]{16,}"),
)


def _reject_secrets(*values: str) -> None:
    combined = "\n".join(values)
    if any(pattern.search(combined) for pattern in _SECRET_PATTERNS):
        raise ValueError("long-term memory must not contain credentials or tokens")


def _metadata(entry: MemoryEntry) -> dict[str, object]:
    return {
        "schema_version": entry.schema_version,
        "memory_id": entry.memory_id,
        "title": entry.title,
        "kind": entry.kind,
        "confidence": entry.confidence,
        "tags": list(entry.tags),
        "source_refs": list(entry.source_refs),
        "source_run_id": entry.source_run_id,
        "created_at": entry.created_at.isoformat(),
        "updated_at": entry.updated_at.isoformat(),
    }


def _render(entry: MemoryEntry) -> str:
    lines = ["---"]
    for key, value in _metadata(entry).items():
        lines.append(f"{key}: {json.dumps(value, ensure_ascii=False)}")
    lines.extend(["---", "", f"# {entry.title}", "", entry.content.strip(), ""])
    return "\n".join(lines)


def _parse(path: Path) -> MemoryEntry:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if len(lines) < 4 or lines[0] != "---":
        raise ValueError(f"invalid memory frontmatter: {path}")
    try:
        boundary = lines.index("---", 1)
    except ValueError as exc:
        raise ValueError(f"unterminated memory frontmatter: {path}") from exc
    metadata: dict[str, object] = {}
    for line in lines[1:boundary]:
        key, separator, raw = line.partition(":")
        if not separator:
            raise ValueError(f"invalid memory metadata line: {path}")
        try:
            metadata[key.strip()] = json.loads(raw.strip())
        except json.JSONDecodeError:
            # Keep simple hand-edited scalar frontmatter readable without
            # adding a general YAML dependency.
            metadata[key.strip()] = raw.strip()
    body = "\n".join(lines[boundary + 1 :]).strip()
    title_prefix = f"# {metadata.get('title', '')}"
    if body.startswith(title_prefix):
        body = body[len(title_prefix) :].strip()
    return MemoryEntry(
        **metadata,
        content=body,
    )


class MarkdownMemoryStore:
    """A small local store rebuilt from Markdown on each recall."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.entries_dir = self.root / "entries"
        self._lock = threading.RLock()

    def save(
        self,
        *,
        title: str,
        content: str,
        kind: MemoryKind,
        confidence: float,
        tags: list[str] | tuple[str, ...] = (),
        source_refs: list[str] | tuple[str, ...] = (),
        source_run_id: str | None = None,
    ) -> MemoryEntry:
        _reject_secrets(title, content, *tags, *source_refs)
        if kind == "finding" and not any(ref.strip() for ref in source_refs):
            raise ValueError("research findings require at least one source reference")
        now = datetime.now(UTC)
        entry = MemoryEntry(
            memory_id=f"mem_{uuid4().hex[:24]}",
            title=title.strip(),
            content=content.strip(),
            kind=kind,
            confidence=confidence,
            tags=tuple(dict.fromkeys(tag.strip() for tag in tags if tag.strip())),
            source_refs=tuple(
                dict.fromkeys(ref.strip() for ref in source_refs if ref.strip())
            ),
            source_run_id=source_run_id,
            created_at=now,
            updated_at=now,
        )
        with self._lock:
            self.entries_dir.mkdir(parents=True, exist_ok=True)
            target = self.entries_dir / f"{entry.memory_id}.md"
            temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
            with temporary.open("w", encoding="utf-8", newline="\n") as handle:
                handle.write(_render(entry))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        return entry

    def list_entries(self) -> tuple[MemoryEntry, ...]:
        if not self.entries_dir.is_dir():
            return ()
        with self._lock:
            entries: list[MemoryEntry] = []
            for path in sorted(self.entries_dir.glob("mem_*.md")):
                try:
                    entries.append(_parse(path))
                except (OSError, ValueError):
                    logger.exception("Skipping unreadable memory entry: %s", path)
        return tuple(sorted(entries, key=lambda entry: entry.updated_at, reverse=True))

    def recall(
        self,
        query: str,
        *,
        limit: int = 5,
        excerpt_chars: int = 1600,
        kinds: set[MemoryKind] | None = None,
    ) -> tuple[MemoryRecall, ...]:
        if limit < 1:
            raise ValueError("memory recall limit must be positive")
        if excerpt_chars < 100:
            raise ValueError("memory excerpt must contain at least 100 characters")
        query_terms = set(tokenize(query))
        if not query_terms:
            return ()
        entries = tuple(
            entry
            for entry in self.list_entries()
            if kinds is None or entry.kind in kinds
        )
        if not entries:
            return ()
        frequencies = tuple(
            Counter(
                tokenize(
                    " ".join(
                        (entry.title, entry.content, *entry.tags, *entry.source_refs)
                    )
                )
            )
            for entry in entries
        )
        lengths = tuple(sum(frequency.values()) for frequency in frequencies)
        average_length = sum(lengths) / max(1, len(lengths))
        document_frequency: Counter[str] = Counter()
        for frequency in frequencies:
            document_frequency.update(frequency.keys())
        idf = {
            term: math.log(
                1 + (len(entries) - count + 0.5) / (count + 0.5)
            )
            for term, count in document_frequency.items()
        }
        scored: list[tuple[float, MemoryEntry]] = []
        for entry, frequency, length in zip(
            entries, frequencies, lengths, strict=True
        ):
            score = 0.0
            for term in query_terms:
                count = frequency.get(term, 0)
                if not count:
                    continue
                denominator = count + 1.5 * (
                    1 - 0.75 + 0.75 * length / max(1.0, average_length)
                )
                score += idf.get(term, 0.0) * count * 2.5 / denominator
            if score > 0:
                scored.append((score, entry))
        scored.sort(key=lambda item: (-item[0], item[1].memory_id))
        return tuple(
            MemoryRecall(
                memory_id=entry.memory_id,
                score=score,
                title=entry.title,
                kind=entry.kind,
                confidence=entry.confidence,
                excerpt=(
                    entry.content
                    if len(entry.content) <= excerpt_chars
                    else entry.content[:excerpt_chars] + " ...[truncated]"
                ),
                tags=entry.tags,
                source_refs=entry.source_refs,
                source_run_id=entry.source_run_id,
            )
            for score, entry in scored[:limit]
        )


__all__ = ["MarkdownMemoryStore"]
