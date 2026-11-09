"""Normalized evidence records for web and personal-knowledge research.

The adapters intentionally retain the unmodified input under ``raw``. The
normalization aliases are explicit and conservative so local source identity
and locators are never guessed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, TypeAlias


class EvidenceSource(str, Enum):
    WEB = "web"
    PERSONAL = "personal"


def _first_text(payload: Mapping[str, Any], aliases: tuple[str, ...]) -> str | None:
    for key in aliases:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return str(value)
    return None


def _json_safe(value: Any) -> Any:
    """Convert retained payloads into deterministic JSON-compatible values."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    return repr(value)


@dataclass(frozen=True, slots=True)
class WebEvidence:
    """Evidence obtained from GPT Researcher's web research path."""

    text: str
    url: str
    title: str | None = None
    evidence_id: str | None = None
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)
    source: EvidenceSource = field(default=EvidenceSource.WEB, init=False)

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("Web evidence text must not be empty")
        if not self.url.strip():
            raise ValueError("Web evidence URL must not be empty")

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "WebEvidence":
        text = _first_text(payload, ("text", "content", "snippet", "excerpt"))
        url = _first_text(payload, ("url", "source_url", "link"))
        if text is None:
            raise ValueError("Web evidence payload has no supported text field")
        if url is None:
            raise ValueError("Web evidence payload has no supported URL field")
        return cls(
            text=text,
            url=url,
            title=_first_text(payload, ("title", "name")),
            evidence_id=_first_text(payload, ("evidence_id", "id")),
            raw=dict(payload),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source.value,
            "text": self.text,
            "url": self.url,
            "title": self.title,
            "evidence_id": self.evidence_id,
            "raw": _json_safe(self.raw),
        }


@dataclass(frozen=True, slots=True)
class PersonalEvidence:
    """Verified evidence returned by embedded private-knowledge retrieval."""

    text: str
    evidence_id: str
    title: str | None = None
    document_id: str | None = None
    task_id: str | None = None
    chunk_id: str | None = None
    source_path: str | None = None
    document_sha256: str | None = None
    release: str | None = None
    version: str | None = None
    clause: str | None = None
    line_start: str | None = None
    line_end: str | None = None
    url: str | None = None
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)
    source: EvidenceSource = field(default=EvidenceSource.PERSONAL, init=False)

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("Personal evidence text must not be empty")
        if not self.evidence_id.strip():
            raise ValueError("Personal evidence ID must not be empty")

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "PersonalEvidence":
        text = _first_text(payload, ("text", "content", "excerpt", "quote"))
        evidence_id = _first_text(payload, ("evidence_id", "id"))
        if text is None:
            raise ValueError("Personal evidence payload has no supported text field")
        if evidence_id is None:
            raise ValueError("Personal evidence payload has no supported evidence ID field")
        return cls(
            text=text,
            evidence_id=evidence_id,
            title=_first_text(
                payload,
                ("title", "document_title", "source_title", "specification"),
            ),
            document_id=_first_text(payload, ("document_id", "doc_id")),
            task_id=_first_text(payload, ("task_id",)),
            chunk_id=_first_text(payload, ("chunk_id",)),
            source_path=_first_text(payload, ("source_path",)),
            document_sha256=_first_text(payload, ("document_sha256",)),
            release=_first_text(payload, ("release", "spec_release")),
            version=_first_text(payload, ("version", "spec_version")),
            clause=_first_text(
                payload,
                ("heading_path", "clause", "section", "section_id"),
            ),
            line_start=_first_text(payload, ("line_start",)),
            line_end=_first_text(payload, ("line_end",)),
            url=_first_text(payload, ("url", "source_url")),
            raw=dict(payload),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source.value,
            "text": self.text,
            "evidence_id": self.evidence_id,
            "title": self.title,
            "document_id": self.document_id,
            "task_id": self.task_id,
            "chunk_id": self.chunk_id,
            "source_path": self.source_path,
            "document_sha256": self.document_sha256,
            "release": self.release,
            "version": self.version,
            "clause": self.clause,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "url": self.url,
            "raw": _json_safe(self.raw),
        }


Evidence: TypeAlias = WebEvidence | PersonalEvidence


def evidence_from_payload(
    source: EvidenceSource | str,
    payload: Mapping[str, Any],
) -> Evidence:
    """Normalize a payload using the parser for its declared source."""

    try:
        normalized_source = source if isinstance(source, EvidenceSource) else EvidenceSource(source)
    except ValueError as exc:
        raise ValueError(f"Unsupported evidence source: {source!r}") from exc
    if normalized_source is EvidenceSource.WEB:
        return WebEvidence.from_payload(payload)
    return PersonalEvidence.from_payload(payload)

