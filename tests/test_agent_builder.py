from types import SimpleNamespace

import deep_agents.builder as builder
from backend.config import Settings
from research_tools.gpt_researcher import CostTracker


def test_builder_disables_general_purpose_and_keeps_only_researcher(monkeypatch, tmp_path):
    captured = {}
    builder._REGISTERED_PROFILES.clear()

    def fake_register(model, profile):
        captured["profile_model"] = model
        captured["profile"] = profile

    def fake_create_deep_agent(**kwargs):
        captured["agent_kwargs"] = kwargs
        return "compiled-graph"

    chief_tool = object()
    researcher_tool = object()
    monkeypatch.setattr(builder, "register_harness_profile", fake_register)
    monkeypatch.setattr(builder, "create_deep_agent", fake_create_deep_agent)
    monkeypatch.setattr(
        builder,
        "build_research_tool_sets",
        lambda settings, mode: SimpleNamespace(
            chief=[chief_tool],
            researcher=[researcher_tool],
            cost_tracker=CostTracker(),
        ),
    )

    built = builder.build_research_agent(
        Settings(deep_agent_model="openai:test-model"),
        source_mode="web",
        run_dir=tmp_path,
    )

    assert built.graph == "compiled-graph"
    assert captured["profile_model"] == "openai:test-model"
    assert captured["profile"].general_purpose_subagent.enabled is False
    kwargs = captured["agent_kwargs"]
    assert kwargs["tools"] == [chief_tool]
    assert [subagent["name"] for subagent in kwargs["subagents"]] == ["researcher"]
    assert kwargs["subagents"][0]["tools"] == [researcher_tool]
    assert "write_todos" in kwargs["system_prompt"]


def test_harness_profile_registration_is_idempotent(monkeypatch):
    calls = []
    builder._REGISTERED_PROFILES.clear()
    monkeypatch.setattr(builder, "register_harness_profile", lambda *args: calls.append(args))

    builder._disable_general_purpose_subagent("openai:test-model")
    builder._disable_general_purpose_subagent("openai:test-model")

    assert len(calls) == 1

