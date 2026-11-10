from deep_agents.prompts import build_chief_prompt, build_researcher_prompt


def test_personal_prompt_does_not_require_web_research():
    prompt = build_researcher_prompt("personal")
    assert "Call `deep_research` exactly once" not in prompt
    assert "search_personal_knowledge" in prompt
    assert "plan, task results, findings" in prompt


def test_personal_chief_delegates_instead_of_running_local_pipeline():
    prompt = build_chief_prompt("personal", max_revision_rounds=0)
    assert "delegate local research to Researchers" in prompt
    assert "Do not run the full local" in prompt
    assert "/sections/<two-digit-index>-<slug>.md" in prompt
    assert "write `/report.md`" in prompt
    assert "never prepend a host path" in prompt


def test_web_prompt_forbids_personal_scope_for_chief():
    prompt = build_chief_prompt("web", max_revision_rounds=1)
    assert "Personal knowledge retrieval is not authorized" in prompt
    assert "at most\n   1 revision" in prompt


def test_mixed_prompt_requires_distinct_source_types():
    prompt = build_researcher_prompt("mixed")
    assert "Call `deep_research` exactly once" in prompt
    assert "Personal Evidence IDs" in prompt
