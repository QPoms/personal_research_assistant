from types import SimpleNamespace

import deep_agents.builder as builder
import pytest
from langchain.agents.middleware import ModelRetryMiddleware
from langchain.agents.middleware import TodoListMiddleware
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from backend.config import Settings
from backend.runtime.working_memory import WorkingMemoryMiddleware
from research_tools.gpt_researcher import CostTracker


def test_builder_disables_general_purpose_and_keeps_only_researcher(
    monkeypatch, tmp_path
):
    captured = {}
    configured_model = object()
    builder._REGISTERED_PROFILES.clear()

    def fake_register(model, profile):
        captured["profile_model"] = model
        captured["profile"] = profile

    def fake_create_deep_agent(**kwargs):
        captured["agent_kwargs"] = kwargs
        return "compiled-graph"

    chief_tool = object()
    researcher_tool = object()
    checkpointer = object()
    chief_context = object()
    researcher_context = object()
    monkeypatch.setattr(builder, "register_harness_profile", fake_register)
    monkeypatch.setattr(builder, "create_deep_agent", fake_create_deep_agent)
    monkeypatch.setattr(
        builder,
        "deep_agent_model",
        lambda model_name, **kwargs: (
            captured.update(model_name=model_name, model_kwargs=kwargs)
            or configured_model
        ),
    )
    monkeypatch.setattr(
        builder,
        "build_research_tool_sets",
        lambda settings, mode, **kwargs: SimpleNamespace(
            chief=[chief_tool],
            researcher=[researcher_tool],
            cost_tracker=CostTracker(),
        ),
    )
    monkeypatch.setattr(
        builder,
        "build_context_middleware_stack",
        lambda settings, model, backend, *, agent, on_compaction=None: [
            chief_context if agent == "chief" else researcher_context
        ],
    )

    built = builder.build_research_agent(
        Settings(
            deep_agent_model="openai:test-model",
            deep_agent_max_output_tokens=1024,
            deep_agent_use_responses_api=False,
        ),
        source_mode="web",
        run_dir=tmp_path,
        checkpointer=checkpointer,
    )

    assert built.graph == "compiled-graph"
    assert captured["profile_model"] == "openai:test-model"
    assert captured["model_name"] == "openai:test-model"
    assert captured["model_kwargs"] == {
        "max_output_tokens": 1024,
        "use_responses_api": False,
    }
    assert captured["profile"].general_purpose_subagent.enabled is False
    kwargs = captured["agent_kwargs"]
    assert kwargs["model"] is configured_model
    assert kwargs["tools"] == [chief_tool]
    assert kwargs["checkpointer"] is checkpointer
    assert [subagent["name"] for subagent in kwargs["subagents"]] == ["researcher"]
    assert kwargs["subagents"][0]["tools"] == [researcher_tool]
    assert kwargs["middleware"][0] is chief_context
    assert kwargs["subagents"][0]["middleware"][0] is researcher_context
    assert isinstance(kwargs["middleware"][1], TodoListMiddleware)
    assert isinstance(kwargs["middleware"][2], WorkingMemoryMiddleware)
    chief_retry = kwargs["middleware"][-1]
    researcher_retry = kwargs["subagents"][0]["middleware"][-1]
    assert isinstance(chief_retry, ModelRetryMiddleware)
    assert isinstance(researcher_retry, ModelRetryMiddleware)
    assert chief_retry is not researcher_retry
    assert chief_retry.max_retries == researcher_retry.max_retries == 5
    assert chief_retry.on_failure == researcher_retry.on_failure == "error"
    assert "write_todos" in kwargs["system_prompt"]
    backend = kwargs["backend"]
    assert backend.virtual_mode is True
    assert backend._resolve_path("/report.md") == tmp_path / "report.md"
    with pytest.raises(ValueError, match="outside root directory"):
        backend._resolve_path(str(tmp_path.parent / "outside.md"))


def test_harness_profile_registration_is_idempotent(monkeypatch):
    calls = []
    builder._REGISTERED_PROFILES.clear()
    monkeypatch.setattr(
        builder, "register_harness_profile", lambda *args: calls.append(args)
    )

    builder._disable_general_purpose_subagent("openai:test-model")
    builder._disable_general_purpose_subagent("openai:test-model")

    assert len(calls) == 1


def test_compiled_chief_exposes_planning_context_and_memory_tools(
    monkeypatch, tmp_path
):
    builder._REGISTERED_PROFILES.clear()
    monkeypatch.setattr(
        builder,
        "deep_agent_model",
        lambda *args, **kwargs: FakeListChatModel(responses=["done"]),
    )
    monkeypatch.setattr(
        builder,
        "build_research_tool_sets",
        lambda *args, **kwargs: SimpleNamespace(
            chief=[], researcher=[], cost_tracker=CostTracker()
        ),
    )

    built = builder.build_research_agent(
        Settings(
            deep_agent_model="openai:test-model",
            long_term_memory_dir=tmp_path / "memory",
        ),
        source_mode="web",
        run_dir=tmp_path / "run",
        run_id="test-run",
    )
    tool_node = built.graph.nodes["tools"].bound
    names = set(tool_node._tools_by_name)

    assert {
        "write_todos",
        "update_working_memory",
        "compact_conversation",
        "recall_long_term_memory",
        "save_long_term_memory",
        "task",
    } <= names
    assert tool_node._injected_args["update_working_memory"].runtime == "runtime"
