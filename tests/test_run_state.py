from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from backend.runtime.run_state import (
    RunRecord,
    RunStage,
    RunStatus,
    transition_run_record,
)


def _record() -> RunRecord:
    return RunRecord(
        run_id="run-1",
        query="topic",
        source_mode="hybrid",
        status=RunStatus.RUNNING,
        stage=RunStage.BUILDING_AGENT,
        started_at=datetime.now(UTC),
        report_path="outputs/run_run-1/report.md",
    )


def test_run_state_follows_forward_only_completion_path():
    record = _record()
    record = transition_run_record(
        record, status=RunStatus.RUNNING, stage=RunStage.RUNNING_AGENT
    )
    record = transition_run_record(
        record, status=RunStatus.RUNNING, stage=RunStage.VALIDATING_REPORT
    )
    finished_at = datetime.now(UTC)
    record = transition_run_record(
        record,
        status=RunStatus.COMPLETED,
        stage=RunStage.COMPLETED,
        finished_at=finished_at,
    )

    assert record.status is RunStatus.COMPLETED
    assert record.finished_at == finished_at


def test_run_state_rejects_backward_and_premature_completion():
    running = transition_run_record(
        _record(), status=RunStatus.RUNNING, stage=RunStage.RUNNING_AGENT
    )

    with pytest.raises(ValueError, match="backwards"):
        transition_run_record(
            running, status=RunStatus.RUNNING, stage=RunStage.BUILDING_AGENT
        )
    with pytest.raises(ValueError, match="only after report validation"):
        transition_run_record(
            running,
            status=RunStatus.COMPLETED,
            stage=RunStage.COMPLETED,
            finished_at=datetime.now(UTC),
        )


def test_terminal_run_cannot_transition_again():
    record = transition_run_record(
        _record(),
        status=RunStatus.CANCELLED,
        stage=RunStage.BUILDING_AGENT,
        finished_at=datetime.now(UTC),
    )

    with pytest.raises(ValueError, match="terminal run"):
        transition_run_record(
            record,
            status=RunStatus.FAILED,
            stage=RunStage.BUILDING_AGENT,
            finished_at=datetime.now(UTC),
            error="late failure",
        )


def test_failed_run_requires_error_and_keeps_active_stage():
    record = _record()

    with pytest.raises(ValueError, match="keeps its active stage"):
        transition_run_record(
            record,
            status=RunStatus.FAILED,
            stage=RunStage.RUNNING_AGENT,
            finished_at=datetime.now(UTC),
            error="boom",
        )
    with pytest.raises(ValidationError, match="requires an error"):
        transition_run_record(
            record,
            status=RunStatus.FAILED,
            stage=RunStage.BUILDING_AGENT,
            finished_at=datetime.now(UTC),
        )
