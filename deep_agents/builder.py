"""Build the Chief Editor harness using Deep Agents' native capabilities.

Adapted from GPT Researcher 0.16.0 ``deep_agents/agent.py``. Todo, filesystem,
and task delegation remain Deep Agents built-ins; this module only adds the
authorized research tools and the explicit Researcher profile.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from deepagents import (
    GeneralPurposeSubagentProfile,
    HarnessProfile,
    create_deep_agent,
    register_harness_profile,
)
from deepagents.backends import FilesystemBackend
from langgraph.checkpoint.base import BaseCheckpointSaver

from backend.config import Settings
from backend.runtime.cancellation import CancellationToken
from backend.runtime.model_factory import deep_agent_model
from backend.runtime.model_retry import retry_policy_from_settings
from deep_agents.prompts import build_chief_prompt, build_researcher_prompt
from research_tools.factory import build_research_tool_sets
from research_tools.gpt_researcher import CostTracker
from research_tools.source_policy import SourceMode, coerce_source_mode


_REGISTERED_PROFILES: set[str] = set()


@dataclass(slots=True)
class BuiltResearchAgent:
    graph: Any
    source_mode: SourceMode
    cost_tracker: CostTracker


def _disable_general_purpose_subagent(model: str) -> None:
    """Keep Deep Agents' task tool but expose only our Researcher subagent."""
    if model in _REGISTERED_PROFILES:
        return
    register_harness_profile(
        model,
        HarnessProfile(
            general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False)
        ),
    )
    _REGISTERED_PROFILES.add(model)


def build_research_agent(
    settings: Settings,
    *,
    source_mode: SourceMode | str,
    run_dir: Path,
    guidelines: list[str] | None = None,
    cancellation_token: CancellationToken | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
) -> BuiltResearchAgent:
    mode = coerce_source_mode(source_mode)
    model_spec = settings.deep_agent_model.strip()
    model = deep_agent_model(
        model_spec,
        max_output_tokens=settings.deep_agent_max_output_tokens,
        use_responses_api=settings.deep_agent_use_responses_api,
    )
    _disable_general_purpose_subagent(model_spec)
    tool_sets = build_research_tool_sets(
        settings, mode, cancellation_token=cancellation_token
    )
    retry_policy = retry_policy_from_settings(settings)

    researcher_subagent = {
        "name": "researcher",
        "description": (
            "Researches one report section using only the run's authorized "
            "sources and writes a citation-preserving Markdown draft."
        ),
        "system_prompt": build_researcher_prompt(mode),
        "tools": tool_sets.researcher,
        "middleware": [retry_policy.middleware()],
    }

    graph = create_deep_agent(
        model=model,
        tools=tool_sets.chief,
        system_prompt=build_chief_prompt(
            mode,
            max_revision_rounds=settings.max_revision_rounds,
            guidelines=guidelines,
        ),
        subagents=[researcher_subagent],
        backend=FilesystemBackend(root_dir=str(run_dir), virtual_mode=True),
        middleware=[retry_policy.middleware()],
        checkpointer=checkpointer,
    )
    return BuiltResearchAgent(
        graph=graph, source_mode=mode, cost_tracker=tool_sets.cost_tracker
    )
