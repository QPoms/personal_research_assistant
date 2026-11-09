import json

import httpx
import pytest

from research_assistant.tools.spec_rag import (
    SpecRAGClient,
    SpecRAGClientConfig,
    SpecRAGError,
    normalise_inspection,
)


def config(**overrides):
    values = {
        "base_url": "https://specrag.test",
        "workspace_id": "workspace-1",
        "access_token": "secret-token",
        "poll_interval_seconds": 0.001,
        "poll_timeout_seconds": 1.0,
    }
    values.update(overrides)
    return SpecRAGClientConfig(**values)


@pytest.mark.asyncio
async def test_create_poll_and_return_completed_verified_evidence():
    inspections = iter(["queued", "running", "completed"])
    submitted = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer secret-token"
        if request.method == "POST":
            submitted.update(json.loads(request.content))
            return httpx.Response(202, json={"run": {"id": "RUN-1", "status": "queued"}})
        status = next(inspections)
        payload = {
            "run": {"id": "RUN-1", "status": status},
            "answer": "Grounded answer" if status == "completed" else None,
            "evidence_ids": ["E001"] if status == "completed" else [],
            "evidence": [
                {
                    "evidence_id": "E001",
                    "specification": "3GPP TS 24.501",
                    "release": 17,
                    "version": "17.10.0",
                    "heading_path": ["5", "Security"],
                    "line_start": 10,
                    "line_end": 14,
                    "excerpt": "Verified text",
                }
            ] if status == "completed" else [],
            "evidence_status": "sufficient" if status == "completed" else None,
            "gaps": [],
        }
        return httpx.Response(200, json=payload)

    client = SpecRAGClient(config(), transport=httpx.MockTransport(handler))
    result = await client.search(query="NAS security")

    assert submitted == {
        "question": "NAS security",
        "release": 17,
        "answer_language": "Chinese",
        "detail_level": "standard",
        "budget": {
            "max_iterations": 4,
            "max_tool_calls": 12,
            "max_gap_tool_calls": 6,
            "max_replans": 3,
        },
        "elapsed_time_seconds": 1200,
    }
    assert result["status"] == "completed"
    assert result["evidence"][0] == {
        "evidence_id": "E001",
        "source_type": "personal",
        "title": "3GPP TS 24.501",
        "locator": "Release 17 / 17.10.0 / Clause 5 > Security / Lines 10-14",
        "excerpt": "Verified text",
    }


@pytest.mark.asyncio
async def test_timeout_returns_pending_and_reusable_run_id():
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(202, json={"run": {"id": "RUN-2"}})
        return httpx.Response(200, json={"run": {"id": "RUN-2", "status": "running"}})

    client = SpecRAGClient(
        config(poll_timeout_seconds=0.003),
        transport=httpx.MockTransport(handler),
    )
    result = await client.search(query="long research")
    assert result["status"] == "pending"
    assert result["run_id"] == "RUN-2"


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal", ["failed", "cancelled", "timed_out"])
async def test_all_real_failure_states_are_terminal(terminal):
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"run": {"id": "RUN-X", "status": terminal}, "failure": {"kind": terminal}},
        )

    client = SpecRAGClient(config(), transport=httpx.MockTransport(handler))
    result = await client.search(run_id="RUN-X")
    assert result["status"] == terminal
    assert result["failure"] == {"kind": terminal}


@pytest.mark.asyncio
async def test_exactly_one_search_argument_is_required():
    client = SpecRAGClient(config())
    with pytest.raises(ValueError, match="exactly one"):
        await client.search()
    with pytest.raises(ValueError, match="exactly one"):
        await client.search(query="x", run_id="RUN-X")


@pytest.mark.asyncio
async def test_post_is_not_automatically_retried_on_http_failure():
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(503, json={"code": "service_unavailable", "detail": "later"})

    client = SpecRAGClient(config(), transport=httpx.MockTransport(handler))
    with pytest.raises(SpecRAGError) as captured:
        await client.create_run("x")
    assert calls == 1
    assert captured.value.retryable is True


def test_completed_partial_evidence_is_preserved_not_promoted_to_sufficient():
    result = normalise_inspection(
        {
            "run": {"id": "RUN-P", "status": "completed"},
            "answer": "Partial answer",
            "evidence_status": "partial",
            "evidence": [],
            "gaps": ["missing clause"],
        }
    )
    assert result["status"] == "completed"
    assert result["evidence_status"] == "partial"
    assert result["unresolved_gaps"] == ["missing clause"]

