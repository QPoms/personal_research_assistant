from research_assistant.agents.prompts import build_chief_prompt, build_researcher_prompt


def test_personal_prompt_does_not_require_web_research():
    prompt = build_researcher_prompt("personal")
    assert "Call `deep_research` exactly once" not in prompt
    assert "search_personal_knowledge" in prompt


def test_web_prompt_forbids_personal_scope_for_chief():
    prompt = build_chief_prompt("web", max_revision_rounds=1)
    assert "SpecRAG/personal knowledge is not authorized" in prompt
    assert "at most\n   1 revision" in prompt


def test_mixed_prompt_requires_distinct_source_types():
    prompt = build_researcher_prompt("mixed")
    assert "Call `deep_research` exactly once" in prompt
    assert "Personal Evidence IDs" in prompt

