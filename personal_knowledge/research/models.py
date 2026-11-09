"""Structured contracts for the slim SpecRAG-derived research workflow."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ResearchTask(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str = Field(pattern=r"^T[1-9][0-9]*$")
    objective: str = Field(min_length=1)
    question: str = Field(min_length=1)
    dependencies: tuple[str, ...] = ()


class ResearchPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tasks: tuple[ResearchTask, ...] = Field(min_length=1, max_length=3)

    @model_validator(mode="after")
    def _dependencies_are_ordered(self) -> "ResearchPlan":
        seen: set[str] = set()
        for expected_index, task in enumerate(self.tasks, 1):
            expected_id = f"T{expected_index}"
            if task.task_id != expected_id:
                raise ValueError(f"task IDs must be ordered as {expected_id}")
            unknown = set(task.dependencies) - seen
            if unknown:
                raise ValueError(
                    f"task {task.task_id} has unknown or forward dependencies: "
                    + ", ".join(sorted(unknown))
                )
            seen.add(task.task_id)
        return self


class TaskResearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str
    objective: str
    status: Literal["sufficient", "partial", "insufficient"]
    supporting_evidence_ids: tuple[str, ...] = ()
    unresolved_gaps: tuple[str, ...] = ()
    retrieval_rounds: int = Field(ge=1, le=2)
    candidate_count: int = Field(ge=0)


class Finding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    finding_id: str = Field(pattern=r"^F[1-9][0-9]*$")
    task_ids: tuple[str, ...] = Field(min_length=1)
    claim: str = Field(min_length=1)
    evidence_ids: tuple[str, ...] = Field(min_length=1)
    status: Literal["supported", "partial"] = "supported"


class LocalResearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["completed", "partial", "unavailable"]
    query: str
    release: int
    plan: ResearchPlan
    task_results: tuple[TaskResearchResult, ...]
    findings: tuple[Finding, ...]
    evidence: tuple[dict[str, Any], ...]
    conflicts: tuple[str, ...] = ()
    unresolved_gaps: tuple[str, ...] = ()
    local_summary: str
    pipeline_warnings: tuple[str, ...] = ()

