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


def _safe_run_id(run_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", run_id):
        raise ValueError("run_id contains unsupported path characters")
    return run_id


def create_run_paths(outputs_dir: Path, run_id: str | None = None) -> RunPaths:
    if run_id is None:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        run_id = f"{timestamp}-{secrets.token_hex(4)}"
    run_id = _safe_run_id(run_id)

    root = outputs_dir.resolve() / f"run_{run_id}"
    sections = root / "sections"
    sections.mkdir(parents=True, exist_ok=False)
    return RunPaths(run_id=run_id, root=root, sections=sections, report=root / "report.md")


