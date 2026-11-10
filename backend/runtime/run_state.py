"""Typed run snapshot and legal state transitions."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RunStatus(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"
    CANCELLED = "cancelled"


class RunStage(StrEnum):
    BUILDING_AGENT = "building_agent"
    RUNNING_AGENT = "running_agent"
    VALIDATING_REPORT = "validating_report"
    COMPLETED = "completed"


_STAGE_ORDER = {
    RunStage.BUILDING_AGENT: 0,
    RunStage.RUNNING_AGENT: 1,
    RunStage.VALIDATING_REPORT: 2,
    RunStage.COMPLETED: 3,
}


class RunRecord(BaseModel):
    """Current durable projection of one research run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["run-record/v1"] = "run-record/v1"
    run_id: str = Field(min_length=1, max_length=128)
    query: str = Field(min_length=1)
    source_mode: str
    status: RunStatus
    stage: RunStage
    started_at: datetime
    finished_at: datetime | None = None
    web_research_cost: float = Field(default=0.0, ge=0)
    report_path: str
    guidelines: tuple[str, ...] = ()
    error: str | None = None

    @model_validator(mode="after")
    def validate_terminal_fields(self) -> "RunRecord":
        if self.status is RunStatus.RUNNING and self.finished_at is not None:
            raise ValueError("running run cannot have finished_at")
        if self.status is not RunStatus.RUNNING and self.finished_at is None:
            raise ValueError("terminal run requires finished_at")
        if self.status is RunStatus.COMPLETED and self.stage is not RunStage.COMPLETED:
            raise ValueError("completed run requires completed stage")
        if self.stage is RunStage.COMPLETED and self.status is not RunStatus.COMPLETED:
            raise ValueError("completed stage requires completed status")
        if self.status is RunStatus.FAILED and not self.error:
            raise ValueError("failed run requires an error")
        return self


def transition_run_record(
    current: RunRecord,
    *,
    status: RunStatus,
    stage: RunStage,
    finished_at: datetime | None = None,
    web_research_cost: float | None = None,
    error: str | None = None,
) -> RunRecord:
    """Create a validated next snapshot without permitting backward transitions."""

    if current.status is RunStatus.INTERRUPTED and status is RunStatus.RUNNING:
        if stage is not current.stage:
            raise ValueError("resumed run keeps its checkpointed stage")
        data = current.model_dump()
        data.update(
            {
                "status": status,
                "finished_at": None,
                "web_research_cost": current.web_research_cost
                if web_research_cost is None
                else web_research_cost,
                "error": None,
            }
        )
        return RunRecord.model_validate(data)

    if current.status is not RunStatus.RUNNING:
        raise ValueError(f"terminal run cannot transition from {current.status.value}")

    if status is RunStatus.RUNNING:
        if stage is RunStage.COMPLETED:
            raise ValueError("running run cannot enter completed stage")
        if _STAGE_ORDER[stage] < _STAGE_ORDER[current.stage]:
            raise ValueError("run stage cannot move backwards")
    elif status is RunStatus.COMPLETED:
        if current.stage is not RunStage.VALIDATING_REPORT:
            raise ValueError("run can complete only after report validation")
        if stage is not RunStage.COMPLETED:
            raise ValueError("completed status requires completed stage")
    elif stage is not current.stage:
        raise ValueError("failed, interrupted, or cancelled run keeps its active stage")

    data = current.model_dump()
    data.update(
        {
            "status": status,
            "stage": stage,
            "finished_at": finished_at,
            "web_research_cost": current.web_research_cost
            if web_research_cost is None
            else web_research_cost,
            "error": error,
        }
    )
    return RunRecord.model_validate(data)


__all__ = ["RunRecord", "RunStage", "RunStatus", "transition_run_record"]
