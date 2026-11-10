"""Small checkpointed working state kept separate from conversation history."""

from __future__ import annotations

import json
from typing import Annotated, NotRequired

from deepagents.middleware._utils import append_to_system_message
from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
    PrivateStateAttr,
)
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.types import Command
from langchain.tools import ToolRuntime
from pydantic import BaseModel, ConfigDict, Field


EvidenceReference = Annotated[str, Field(min_length=1, max_length=300)]
WorkingMemoryItem = Annotated[str, Field(min_length=1, max_length=1000)]
ArtifactReference = Annotated[str, Field(min_length=1, max_length=500)]


class WorkingMemorySnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    objective: str = Field(min_length=1, max_length=2000)
    current_focus: str = Field(default="", max_length=2000)
    selected_evidence_ids: list[EvidenceReference] = Field(
        default_factory=list, max_length=40
    )
    open_questions: list[WorkingMemoryItem] = Field(
        default_factory=list, max_length=20
    )
    decisions: list[WorkingMemoryItem] = Field(default_factory=list, max_length=20)
    artifact_paths: list[ArtifactReference] = Field(
        default_factory=list, max_length=40
    )


class WorkingMemoryState(AgentState):
    working_memory: Annotated[
        NotRequired[dict[str, object]],
        PrivateStateAttr,
    ]


class UpdateWorkingMemoryInput(WorkingMemorySnapshot):
    """Replacement snapshot; callers send the complete current state."""


@tool("update_working_memory", args_schema=UpdateWorkingMemoryInput)
def update_working_memory(
    objective: str,
    runtime: ToolRuntime,
    current_focus: str = "",
    selected_evidence_ids: list[str] | None = None,
    open_questions: list[str] | None = None,
    decisions: list[str] | None = None,
    artifact_paths: list[str] | None = None,
) -> Command:
    """Replace compact task state without copying raw evidence or tool results."""

    snapshot = WorkingMemorySnapshot(
        objective=objective,
        current_focus=current_focus,
        selected_evidence_ids=selected_evidence_ids or [],
        open_questions=open_questions or [],
        decisions=decisions or [],
        artifact_paths=artifact_paths or [],
    )
    tool_call_id = runtime.tool_call_id or "working-memory"
    return Command(
        update={
            "working_memory": snapshot.model_dump(mode="json"),
            "messages": [
                ToolMessage(
                    content=(
                        "Working memory updated: "
                        f"{len(snapshot.selected_evidence_ids)} evidence references, "
                        f"{len(snapshot.open_questions)} open questions, "
                        f"{len(snapshot.artifact_paths)} artifacts."
                    ),
                    tool_call_id=tool_call_id,
                )
            ],
        }
    )


_WORKING_MEMORY_PROMPT = """<working_memory>
{working_memory}
</working_memory>

Use `update_working_memory` after planning and whenever task status, selected
evidence, unresolved questions, decisions, or artifact paths materially change.
Keep only compact references and state here. Never copy raw documents, long tool
results, report drafts, credentials, or full evidence excerpts into working memory.
"""


class WorkingMemoryMiddleware(AgentMiddleware):
    state_schema = WorkingMemoryState
    tools = [update_working_memory]

    def _prompt(self, request: ModelRequest):
        memory = request.state.get("working_memory") or {}
        rendered = _WORKING_MEMORY_PROMPT.format(
            working_memory=json.dumps(memory, ensure_ascii=False, separators=(",", ":"))
        )
        return request.override(
            system_message=append_to_system_message(request.system_message, rendered)
        )

    def wrap_model_call(self, request: ModelRequest, handler) -> ModelResponse:
        return handler(self._prompt(request))

    async def awrap_model_call(self, request: ModelRequest, handler) -> ModelResponse:
        return await handler(self._prompt(request))


def initial_working_memory(objective: str) -> dict[str, object]:
    return WorkingMemorySnapshot(objective=objective).model_dump(mode="json")


__all__ = [
    "WorkingMemoryMiddleware",
    "WorkingMemorySnapshot",
    "WorkingMemoryState",
    "initial_working_memory",
    "update_working_memory",
]
