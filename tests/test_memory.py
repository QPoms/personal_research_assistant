from __future__ import annotations

import pytest

from backend.memory.middleware import LongTermMemoryMiddleware
from backend.memory.store import MarkdownMemoryStore


def test_markdown_memory_round_trips_and_bm25_recalls(tmp_path):
    store = MarkdownMemoryStore(tmp_path / "memory")
    timer = store.save(
        title="3GPP NAS timer finding",
        content="TS 24.501 evidence showed the NAS registration timer behavior.",
        kind="finding",
        confidence=0.9,
        tags=["3gpp", "NAS"],
        source_refs=["evidence:ev-24-501"],
        source_run_id="run-1",
    )
    store.save(
        title="Report formatting preference",
        content="Prefer concise comparison tables in final reports.",
        kind="preference",
        confidence=1.0,
        tags=["formatting"],
        source_run_id="run-2",
    )

    memory_path = tmp_path / "memory" / "entries" / f"{timer.memory_id}.md"
    text = memory_path.read_text(encoding="utf-8")
    assert text.startswith("---\n")
    assert "# 3GPP NAS timer finding" in text
    assert MarkdownMemoryStore(tmp_path / "memory").list_entries()[0].source_run_id

    recalled = store.recall("What is the 3GPP NAS registration timer?", limit=1)
    assert len(recalled) == 1
    assert recalled[0].memory_id == timer.memory_id
    assert recalled[0].source_refs == ("evidence:ev-24-501",)


def test_memory_rejects_credentials_and_skips_corrupt_entry(tmp_path):
    store = MarkdownMemoryStore(tmp_path / "memory")
    with pytest.raises(ValueError, match="credentials"):
        store.save(
            title="provider configuration",
            content="api_key=sk-1234567890abcdefghijklmnop",
            kind="procedure",
            confidence=1.0,
        )

    entries = tmp_path / "memory" / "entries"
    entries.mkdir(parents=True)
    (entries / "mem_aaaaaaaaaaaaaaaaaaaaaaaa.md").write_text(
        "not valid frontmatter", encoding="utf-8"
    )
    assert store.list_entries() == ()


def test_long_term_memory_middleware_exposes_only_bounded_tools(tmp_path):
    middleware = LongTermMemoryMiddleware(
        MarkdownMemoryStore(tmp_path / "memory"),
        source_run_id="run-1",
        max_results=3,
        excerpt_chars=500,
    )

    assert [tool.name for tool in middleware.tools] == [
        "recall_long_term_memory",
        "save_long_term_memory",
    ]
    assert middleware.tools[0].args_schema.model_fields["limit"].metadata

    saved = middleware.tools[1].invoke(
        {
            "title": "Verified protocol finding",
            "content": "The result was verified against the selected clause.",
            "kind": "finding",
            "confidence": 0.8,
            "tags": ["protocol"],
            "source_refs": ["evidence:ev-1"],
        }
    )
    recalled = middleware.tools[0].invoke({"query": "selected protocol clause"})
    assert saved["saved"] is True
    assert recalled[0]["memory_id"] == saved["memory_id"]


def test_research_finding_requires_source_reference(tmp_path):
    store = MarkdownMemoryStore(tmp_path / "memory")
    with pytest.raises(ValueError, match="source reference"):
        store.save(
            title="Unsupported finding",
            content="A model-generated statement without provenance.",
            kind="finding",
            confidence=0.5,
        )
