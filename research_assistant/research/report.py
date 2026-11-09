"""Lightweight report validation; citation generation remains engine-owned."""

from __future__ import annotations

from pathlib import Path


def validate_report_file(path: Path) -> list[str]:
    """Return structural warnings without rewriting citations or content."""
    if not path.is_file():
        return ["report file does not exist"]
    text = path.read_text(encoding="utf-8")
    warnings: list[str] = []
    if not text.lstrip().startswith("#"):
        warnings.append("report has no Markdown title")
    if "References" not in text and "参考" not in text:
        warnings.append("report has no references heading")
    if len(text.strip()) < 200:
        warnings.append("report is unexpectedly short")
    return warnings


