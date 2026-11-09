"""Command-line entry point for Personal Research Assistant."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
import sys

from backend.config import load_settings
from backend.research.report import validate_report_file
from backend.research.run_manager import run_research
from research_tools.source_policy import SourceMode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an evidence-grounded research task")
    parser.add_argument("--query", required=True, help="Research question or topic")
    parser.add_argument(
        "--source-mode",
        choices=[mode.value for mode in SourceMode],
        default=SourceMode.MIXED.value,
        help="Authorized knowledge sources",
    )
    parser.add_argument(
        "--guideline",
        action="append",
        default=[],
        help="Additional report requirement; may be repeated",
    )
    parser.add_argument("--run-id", help="Optional caller-provided run identifier")
    return parser


async def _async_main(args: argparse.Namespace) -> int:
    settings = load_settings()
    result = await run_research(
        args.query,
        args.source_mode,
        settings,
        guidelines=args.guideline,
        run_id=args.run_id,
    )
    warnings = validate_report_file(result.report_path)
    print(result.summary)
    print(f"Report: {result.report_path}")
    print(f"Run: {result.run_id}")
    print(f"Web research cost: {result.web_research_cost:.6f}")
    for warning in warnings:
        print(f"Warning: {warning}", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return asyncio.run(_async_main(args))
    except (ValueError, RuntimeError) as exc:
        print(f"Configuration or runtime error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        # Provider and network failures should be recorded in run.json by the
        # run manager and reported without leaking a full traceback or secrets.
        print(f"Research failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Research interrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())


