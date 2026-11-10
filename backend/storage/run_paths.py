"""Create isolated output directories for research runs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
import re
import secrets


@dataclass(frozen=True, slots=True)
class RunPaths:
    run_id: str
    root: Path
    sections: Path
    report: Path
    events: Path


def _safe_run_id(run_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", run_id):
        raise ValueError("run_id contains unsupported path characters")
    return run_id


def generate_run_id() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{secrets.token_hex(4)}"


def create_run_paths(outputs_dir: Path, run_id: str | None = None) -> RunPaths:
    if run_id is None:
        run_id = generate_run_id()
    run_id = _safe_run_id(run_id)

    root = outputs_dir.resolve() / f"run_{run_id}"
    sections = root / "sections"
    sections.mkdir(parents=True, exist_ok=False)
    return RunPaths(
        run_id=run_id,
        root=root,
        sections=sections,
        report=root / "report.md",
        events=root / "events.jsonl",
    )


def load_run_paths(outputs_dir: Path, run_id: str) -> RunPaths:
    run_id = _safe_run_id(run_id)
    root = outputs_dir.resolve() / f"run_{run_id}"
    sections = root / "sections"
    if not root.is_dir() or not sections.is_dir():
        raise FileNotFoundError(f"run output directory is unavailable: {run_id}")
    return RunPaths(
        run_id=run_id,
        root=root,
        sections=sections,
        report=root / "report.md",
        events=root / "events.jsonl",
    )


__all__ = ["RunPaths", "create_run_paths", "generate_run_id", "load_run_paths"]
