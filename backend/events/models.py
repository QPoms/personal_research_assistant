"""Versioned event contract shared by runtime, API, and future frontend."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, JsonValue


class RunEventType(StrEnum):
    RUN_CREATED = "run.created"
    RUN_STAGE_CHANGED = "run.stage_changed"
    RUN_COMPLETED = "run.completed"
    RUN_FAILED = "run.failed"
    RUN_INTERRUPTED = "run.interrupted"
    RUN_CANCELLED = "run.cancelled"
    RUN_RESUMED = "run.resumed"
    PLAN_CREATED = "plan.created"
    TASK_STARTED = "task.started"
    TASK_COMPLETED = "task.completed"
    MODEL_STARTED = "model.started"
    MODEL_RETRYING = "model.retrying"
    MODEL_COMPLETED = "model.completed"
    TOOL_STARTED = "tool.started"
    TOOL_COMPLETED = "tool.completed"
    TOOL_FAILED = "tool.failed"
    EVIDENCE_SELECTED = "evidence.selected"
    CONTEXT_COMPACTED = "context.compacted"
    REPORT_UPDATED = "report.updated"


class RunEvent(BaseModel):
    """One immutable event in a run-scoped, monotonically ordered stream."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["run-event/v1"] = "run-event/v1"
    event_id: str = Field(default_factory=lambda: f"evt_{uuid4().hex}")
    seq: int = Field(ge=1)
    run_id: str = Field(min_length=1, max_length=128)
    event_type: RunEventType
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    source: str = Field(default="run_manager", min_length=1, max_length=128)
    status: str | None = None
    stage: str | None = None
    agent: str | None = None
    task_id: str | None = None
    payload: dict[str, JsonValue] = Field(default_factory=dict)


__all__ = ["RunEvent", "RunEventType"]
