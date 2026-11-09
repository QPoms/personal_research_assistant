"""Program-level source and research-tool authorization.

The language model is deliberately not trusted to enforce source boundaries.
Callers build the Chief Editor and Researcher with only the tools returned by
this module.  Deep Agents' built-in tools (todo, filesystem, and delegation)
are outside this custom research-tool allowlist.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, TypeVar


QUICK_SEARCH = "quick_search"
DEEP_RESEARCH = "deep_research"
PERSONAL_KNOWLEDGE_SEARCH = "search_personal_knowledge"


class SourceMode(str, Enum):
    """Knowledge sources authorized for one research run."""

    WEB = "web"
    PERSONAL = "personal"
    MIXED = "mixed"


class AgentRole(str, Enum):
    """Agent roles whose custom research tools have distinct privileges."""

    CHIEF = "chief"
    RESEARCHER = "researcher"


class ToolAuthorizationError(ValueError):
    """Raised when code attempts to expose an unauthorized research tool."""


_TOOL_ALLOWLIST: dict[tuple[AgentRole, SourceMode], tuple[str, ...]] = {
    (AgentRole.CHIEF, SourceMode.WEB): (QUICK_SEARCH,),
    (AgentRole.CHIEF, SourceMode.PERSONAL): (PERSONAL_KNOWLEDGE_SEARCH,),
    (AgentRole.CHIEF, SourceMode.MIXED): (
        QUICK_SEARCH,
        PERSONAL_KNOWLEDGE_SEARCH,
    ),
    (AgentRole.RESEARCHER, SourceMode.WEB): (
        QUICK_SEARCH,
        DEEP_RESEARCH,
    ),
    (AgentRole.RESEARCHER, SourceMode.PERSONAL): (
        PERSONAL_KNOWLEDGE_SEARCH,
    ),
    (AgentRole.RESEARCHER, SourceMode.MIXED): (
        QUICK_SEARCH,
        DEEP_RESEARCH,
        PERSONAL_KNOWLEDGE_SEARCH,
    ),
}


def coerce_source_mode(value: SourceMode | str) -> SourceMode:
    """Return a validated source mode, accepting CLI-style strings."""

    if isinstance(value, SourceMode):
        return value
    try:
        return SourceMode(value.strip().lower())
    except (AttributeError, ValueError) as exc:
        choices = ", ".join(mode.value for mode in SourceMode)
        raise ValueError(f"Invalid source mode {value!r}; expected one of: {choices}") from exc


def coerce_agent_role(value: AgentRole | str) -> AgentRole:
    """Return a validated agent role."""

    if isinstance(value, AgentRole):
        return value
    try:
        return AgentRole(value.strip().lower())
    except (AttributeError, ValueError) as exc:
        choices = ", ".join(role.value for role in AgentRole)
        raise ValueError(f"Invalid agent role {value!r}; expected one of: {choices}") from exc


@dataclass(frozen=True, slots=True)
class SourcePolicy:
    """Immutable authorization policy for an agent in a single run."""

    mode: SourceMode
    role: AgentRole

    @classmethod
    def from_values(
        cls,
        mode: SourceMode | str,
        role: AgentRole | str,
    ) -> "SourcePolicy":
        return cls(mode=coerce_source_mode(mode), role=coerce_agent_role(role))

    @property
    def allowed_tool_names(self) -> tuple[str, ...]:
        return _TOOL_ALLOWLIST[(self.role, self.mode)]

    def allows(self, tool_name: str) -> bool:
        return tool_name in self.allowed_tool_names

    def require_allowed(self, tool_name: str) -> None:
        if not self.allows(tool_name):
            raise ToolAuthorizationError(
                f"Tool {tool_name!r} is not authorized for "
                f"role={self.role.value!r}, source_mode={self.mode.value!r}"
            )


def allowed_tool_names(
    mode: SourceMode | str,
    role: AgentRole | str,
) -> tuple[str, ...]:
    """Return the ordered custom-tool allowlist for an agent."""

    return SourcePolicy.from_values(mode, role).allowed_tool_names


_ToolT = TypeVar("_ToolT")


def select_allowed_tools(
    available_tools: Mapping[str, _ToolT],
    mode: SourceMode | str,
    role: AgentRole | str,
    *,
    require_complete: bool = True,
) -> list[_ToolT]:
    """Select tools from a registry without ever passing through extras.

    ``require_complete`` defaults to true so a misconfigured agent fails at
    construction time rather than silently running without an expected source.
    """

    names = allowed_tool_names(mode, role)
    missing = [name for name in names if name not in available_tools]
    if missing and require_complete:
        raise KeyError(f"Missing required research tools: {', '.join(missing)}")
    return [available_tools[name] for name in names if name in available_tools]
