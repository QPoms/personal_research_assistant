"""Ordinary HTTP/LangChain Tool adapter for the independent SpecRAG service."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Mapping
from urllib.parse import quote

import httpx
from langchain_core.tools import tool

from research_assistant.config import Settings


class SpecRAGError(RuntimeError):
    """A typed, user-safe SpecRAG integration failure."""

    def __init__(self, message: str, *, category: str, retryable: bool = False) -> None:
        super().__init__(message)
        self.category = category
        self.retryable = retryable


@dataclass(frozen=True, slots=True)
class SpecRAGClientConfig:
    base_url: str
    workspace_id: str
    access_token: str
    release: int = 17
    answer_language: str = "Chinese"
    detail_level: str = "standard"
    poll_interval_seconds: float = 2.0
    poll_timeout_seconds: float = 60.0
    http_timeout_seconds: float = 30.0


_COMPLETED = {"completed", "complete", "succeeded", "success", "done"}
_FAILED = {"failed", "failure", "cancelled", "canceled", "timed_out", "error"}


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _run_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    nested = payload.get("run")
    return nested if isinstance(nested, Mapping) else payload


def _run_id(payload: Mapping[str, Any]) -> str:
    run = _run_payload(payload)
    value = payload.get("run_id") or run.get("run_id") or run.get("id") or payload.get("id")
    if not value:
        raise SpecRAGError(
            "SpecRAG accepted a run but did not return a run identifier",
            category="invalid_response",
        )
    return str(value)


def _status(payload: Mapping[str, Any]) -> str:
    run = _run_payload(payload)
    value = run.get("status") or payload.get("status") or "unknown"
    return str(value).lower()


def _locator(evidence: Mapping[str, Any]) -> str:
    explicit = evidence.get("locator")
    if explicit:
        return str(explicit)
    pieces: list[str] = []
    release = evidence.get("release") or evidence.get("spec_release")
    version = evidence.get("version") or evidence.get("spec_version")
    clause = (
        evidence.get("heading_path")
        or evidence.get("clause")
        or evidence.get("section")
        or evidence.get("section_id")
    )
    line_start = evidence.get("line_start")
    line_end = evidence.get("line_end")
    lines = evidence.get("line_range") or evidence.get("lines")
    if release is not None:
        pieces.append(f"Release {release}")
    if version:
        pieces.append(str(version))
    if clause:
        if isinstance(clause, (list, tuple)):
            clause = " > ".join(str(part) for part in clause)
        pieces.append(f"Clause {clause}")
    if line_start is not None:
        lines = f"{line_start}-{line_end}" if line_end is not None else str(line_start)
    if lines:
        pieces.append(f"Lines {lines}")
    return " / ".join(pieces)


def _normalise_evidence(item: Any, fallback_id: Any = None) -> dict[str, Any] | None:
    evidence = _mapping(item)
    if not evidence:
        return None
    evidence_id = evidence.get("evidence_id") or evidence.get("id") or fallback_id
    if evidence_id is None:
        return None
    return {
        "evidence_id": str(evidence_id),
        "source_type": "personal",
        "title": str(
            evidence.get("title")
            or evidence.get("document_title")
            or evidence.get("specification")
            or evidence.get("source")
            or ""
        ),
        "locator": _locator(evidence),
        "excerpt": str(
            evidence.get("excerpt")
            or evidence.get("quote")
            or evidence.get("text")
            or evidence.get("content")
            or ""
        ),
    }


def normalise_inspection(payload: Mapping[str, Any], *, run_id: str | None = None) -> dict[str, Any]:
    """Keep the real SpecRAG status while exposing only citation-safe fields."""
    resolved_run_id = run_id or _run_id(payload)
    status = _status(payload)
    raw_evidence = payload.get("evidence") or []
    evidence_ids = payload.get("evidence_ids") or []
    evidence: list[dict[str, Any]] = []
    if isinstance(raw_evidence, Mapping):
        for key, value in raw_evidence.items():
            normalised = _normalise_evidence(value, key)
            if normalised:
                evidence.append(normalised)
    elif isinstance(raw_evidence, list):
        for index, value in enumerate(raw_evidence):
            fallback = evidence_ids[index] if index < len(evidence_ids) else None
            normalised = _normalise_evidence(value, fallback)
            if normalised:
                evidence.append(normalised)

    return {
        "status": status,
        "run_id": resolved_run_id,
        "answer": payload.get("answer"),
        "evidence": evidence,
        "evidence_status": payload.get("evidence_status"),
        "unresolved_gaps": payload.get("gaps") or [],
        "failure": payload.get("failure"),
    }


class SpecRAGClient:
    """Minimal async client for the two existing SpecRAG run endpoints."""

    def __init__(
        self,
        config: SpecRAGClientConfig,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.config = config
        self._transport = transport

    @property
    def runs_url(self) -> str:
        workspace = quote(self.config.workspace_id, safe="")
        return f"/api/v1/workspaces/{workspace}/runs"

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self.config.base_url,
            headers={
                "Authorization": f"Bearer {self.config.access_token}",
                "Accept": "application/json",
            },
            timeout=self.config.http_timeout_seconds,
            transport=self._transport,
        )

    async def create_run(self, query: str) -> str:
        request = {
            "question": query,
            "release": self.config.release,
            "answer_language": self.config.answer_language,
            "detail_level": self.config.detail_level,
            "budget": {
                "max_iterations": 4,
                "max_tool_calls": 12,
                "max_gap_tool_calls": 6,
                "max_replans": 3,
            },
            "elapsed_time_seconds": 1200,
        }
        try:
            async with self._client() as client:
                response = await client.post(self.runs_url, json=request)
            if response.status_code != 202:
                detail = _mapping(response.json()) if response.content else {}
                raise SpecRAGError(
                    "SpecRAG run submission returned "
                    f"HTTP {response.status_code}: {detail.get('code') or detail.get('detail') or 'request rejected'}",
                    category="http_error",
                    retryable=response.status_code in {429, 502, 503, 504},
                )
            return _run_id(_mapping(response.json()))
        except SpecRAGError:
            raise
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            raise SpecRAGError(
                "SpecRAG run submission could not reach the service",
                category="unavailable",
                retryable=True,
            ) from exc
        except (ValueError, TypeError) as exc:
            raise SpecRAGError(
                "SpecRAG returned an invalid submission response",
                category="invalid_response",
            ) from exc

    async def inspect_run(self, run_id: str) -> dict[str, Any]:
        if not run_id or "/" in run_id or "\\" in run_id:
            raise ValueError("run_id must be a non-empty opaque identifier")
        try:
            async with self._client() as client:
                response = await client.get(f"{self.runs_url}/{run_id}")
            if response.status_code != 200:
                detail = _mapping(response.json()) if response.content else {}
                raise SpecRAGError(
                    "SpecRAG run inspection returned "
                    f"HTTP {response.status_code}: {detail.get('code') or detail.get('detail') or 'request rejected'}",
                    category="http_error",
                    retryable=response.status_code in {429, 502, 503, 504},
                )
            return normalise_inspection(_mapping(response.json()), run_id=run_id)
        except SpecRAGError:
            raise
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            raise SpecRAGError(
                "SpecRAG run inspection could not reach the service",
                category="unavailable",
                retryable=True,
            ) from exc
        except (ValueError, TypeError) as exc:
            raise SpecRAGError(
                "SpecRAG returned an invalid inspection response",
                category="invalid_response",
            ) from exc

    async def search(self, *, query: str | None = None, run_id: str | None = None) -> dict[str, Any]:
        if (query is None) == (run_id is None):
            raise ValueError("Provide exactly one of query or run_id")
        if query is not None:
            query = query.strip()
            if not query:
                raise ValueError("query must not be empty")
            run_id = await self.create_run(query)

        assert run_id is not None
        if self.config.poll_timeout_seconds == 0:
            return await self.inspect_run(run_id)

        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.config.poll_timeout_seconds
        while True:
            result = await self.inspect_run(run_id)
            if result["status"] in _COMPLETED | _FAILED:
                return result
            remaining = deadline - loop.time()
            if remaining <= 0:
                return {**result, "status": "pending"}
            await asyncio.sleep(min(self.config.poll_interval_seconds, remaining))


def _config_from_settings(settings: Settings) -> SpecRAGClientConfig:
    settings.validate_for_personal_research()
    assert settings.specrag_workspace_id is not None
    assert settings.specrag_access_token is not None
    return SpecRAGClientConfig(
        base_url=settings.specrag_base_url,
        workspace_id=settings.specrag_workspace_id,
        access_token=settings.specrag_access_token.get_secret_value(),
        release=settings.specrag_release,
        answer_language=settings.specrag_answer_language,
        detail_level=settings.specrag_detail_level,
        poll_interval_seconds=settings.specrag_poll_interval_seconds,
        poll_timeout_seconds=settings.specrag_poll_timeout_seconds,
        http_timeout_seconds=settings.specrag_http_timeout_seconds,
    )


def build_spec_rag_tool(settings: Settings):
    """Build the ordinary LangChain tool with credentials closed over."""
    client = SpecRAGClient(_config_from_settings(settings))

    @tool("search_personal_knowledge")
    async def search_personal_knowledge(
        query: str | None = None,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        """Research the configured private SpecRAG workspace.

        Provide exactly one of ``query`` (start and poll a new run) or
        ``run_id`` (inspect an existing run). Workspace and credentials are
        fixed by application configuration and cannot be selected by the model.
        """

        try:
            return await client.search(query=query, run_id=run_id)
        except SpecRAGError as exc:
            return {
                "status": "error",
                "run_id": run_id,
                "answer": None,
                "evidence": [],
                "unresolved_gaps": [],
                "failure": {
                    "category": exc.category,
                    "message": str(exc),
                    "retryable": exc.retryable,
                },
            }

    return search_personal_knowledge


