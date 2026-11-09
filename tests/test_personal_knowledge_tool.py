from pathlib import Path

import pytest

from backend.config import Settings
from research_tools.personal_knowledge import build_personal_knowledge_tool


def _corpus(root: Path) -> None:
    source = root / "data" / "3gpp" / "24.501" / "v17.10.0" / "24501-ha0.md"
    source.parent.mkdir(parents=True)
    source.write_text(
        "# 3GPP TS 24.501\n\n"
        "## 5.4 NAS security\n\n"
        "The UE shall apply integrity protection to protected NAS messages.\n\n"
        "Ciphering protects the confidentiality of NAS signalling.\n",
        encoding="utf-8",
    )


@pytest.mark.asyncio
async def test_embedded_tool_returns_source_verified_evidence(tmp_path, monkeypatch):
    _corpus(tmp_path)
    monkeypatch.setenv("PERSONAL_KNOWLEDGE_ROOT", str(tmp_path))
    settings = Settings(
        personal_knowledge_root=tmp_path,
        personal_knowledge_allow_index_build=True,
        personal_knowledge_max_results=3,
        personal_knowledge_research_enabled=False,
    )
    tool = build_personal_knowledge_tool(settings)

    result = await tool.ainvoke({"query": "NAS integrity protection"})

    assert result["status"] in {"completed", "partial"}
    assert result["evidence"]
    evidence = result["evidence"][0]
    assert evidence["source_valid"] is True
    assert evidence["specification"] == "24.501"
    assert evidence["release"] == 17
    assert evidence["version"] == "17.10.0"
    assert "NAS security" in evidence["locator"]
    assert evidence["document_sha256"]


def test_personal_mode_requires_existing_local_corpus(tmp_path):
    settings = Settings(personal_knowledge_root=tmp_path)
    with pytest.raises(ValueError, match="corpus does not exist"):
        settings.validate_for_personal_research()


def test_full_local_research_requires_explicit_external_model_authorization(tmp_path):
    _corpus(tmp_path)
    settings = Settings(
        personal_knowledge_root=tmp_path,
        personal_knowledge_allow_index_build=True,
        personal_knowledge_research_enabled=True,
        personal_knowledge_allow_external_model=False,
    )
    with pytest.raises(ValueError, match="send selected private evidence"):
        build_personal_knowledge_tool(settings)


@pytest.mark.asyncio
async def test_query_is_the_only_model_controlled_argument(tmp_path, monkeypatch):
    _corpus(tmp_path)
    monkeypatch.setenv("PERSONAL_KNOWLEDGE_ROOT", str(tmp_path))
    tool = build_personal_knowledge_tool(
        Settings(
            personal_knowledge_root=tmp_path,
            personal_knowledge_allow_index_build=True,
            personal_knowledge_research_enabled=False,
        )
    )
    schema = tool.args_schema.model_json_schema()
    assert set(schema["properties"]) == {"query"}
