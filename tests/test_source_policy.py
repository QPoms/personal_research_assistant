import pytest

from research_assistant.research.source_policy import (
    DEEP_RESEARCH,
    PERSONAL_KNOWLEDGE_SEARCH,
    QUICK_SEARCH,
    AgentRole,
    SourceMode,
    SourcePolicy,
    ToolAuthorizationError,
    allowed_tool_names,
    select_allowed_tools,
)


@pytest.mark.parametrize(
    ("role", "mode", "expected"),
    [
        (AgentRole.CHIEF, SourceMode.WEB, (QUICK_SEARCH,)),
        (AgentRole.CHIEF, SourceMode.PERSONAL, (PERSONAL_KNOWLEDGE_SEARCH,)),
        (
            AgentRole.CHIEF,
            SourceMode.MIXED,
            (QUICK_SEARCH, PERSONAL_KNOWLEDGE_SEARCH),
        ),
        (
            AgentRole.RESEARCHER,
            SourceMode.WEB,
            (QUICK_SEARCH, DEEP_RESEARCH),
        ),
        (
            AgentRole.RESEARCHER,
            SourceMode.PERSONAL,
            (PERSONAL_KNOWLEDGE_SEARCH,),
        ),
        (
            AgentRole.RESEARCHER,
            SourceMode.MIXED,
            (QUICK_SEARCH, DEEP_RESEARCH, PERSONAL_KNOWLEDGE_SEARCH),
        ),
    ],
)
def test_role_and_mode_have_exact_allowlist(role, mode, expected):
    assert allowed_tool_names(mode, role) == expected


def test_string_values_are_normalized_for_cli_usage():
    policy = SourcePolicy.from_values(" MIXED ", "RESEARCHER")
    assert policy.mode is SourceMode.MIXED
    assert policy.role is AgentRole.RESEARCHER


def test_chief_cannot_call_deep_research_even_in_mixed_mode():
    policy = SourcePolicy.from_values(SourceMode.MIXED, AgentRole.CHIEF)
    assert not policy.allows(DEEP_RESEARCH)
    with pytest.raises(ToolAuthorizationError, match="not authorized"):
        policy.require_allowed(DEEP_RESEARCH)


def test_personal_mode_does_not_expose_any_web_tool():
    registry = {
        QUICK_SEARCH: object(),
        DEEP_RESEARCH: object(),
        PERSONAL_KNOWLEDGE_SEARCH: object(),
        "unexpected_tool": object(),
    }
    selected = select_allowed_tools(registry, "personal", "researcher")
    assert selected == [registry[PERSONAL_KNOWLEDGE_SEARCH]]


def test_registry_extras_never_pass_through():
    registry = {
        QUICK_SEARCH: "quick",
        PERSONAL_KNOWLEDGE_SEARCH: "personal",
        "admin_secret_tool": "must-not-leak",
    }
    assert select_allowed_tools(registry, "mixed", "chief") == ["quick", "personal"]


def test_missing_authorized_tool_fails_agent_construction_by_default():
    with pytest.raises(KeyError, match=PERSONAL_KNOWLEDGE_SEARCH):
        select_allowed_tools({QUICK_SEARCH: object()}, "mixed", "chief")


def test_invalid_policy_values_fail_closed():
    with pytest.raises(ValueError, match="Invalid source mode"):
        SourcePolicy.from_values("all", "chief")
    with pytest.raises(ValueError, match="Invalid agent role"):
        SourcePolicy.from_values("web", "reviewer")
