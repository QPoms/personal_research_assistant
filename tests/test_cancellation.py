from concurrent.futures import ThreadPoolExecutor
import asyncio

import pytest

from backend.runtime.cancellation import (
    CancellationToken,
    RunCancelledError,
    await_with_cancellation,
)
from research_tools.gpt_researcher import build_web_research_tools


def test_cancellation_token_is_idempotent_and_interrupts_blocking_sleep():
    token = CancellationToken()

    with ThreadPoolExecutor(max_workers=1) as executor:
        sleeper = executor.submit(token.sleep, 60)
        assert token.cancel("stopped by user") is True
        assert token.cancel("second reason") is False
        with pytest.raises(RunCancelledError, match="stopped by user"):
            sleeper.result()

    assert token.is_cancelled is True
    assert token.reason == "stopped by user"


@pytest.mark.asyncio
async def test_await_with_cancellation_stops_async_operation():
    token = CancellationToken()
    started = asyncio.Event()
    cleaned_up = asyncio.Event()

    async def operation():
        started.set()
        try:
            await asyncio.Future()
        finally:
            cleaned_up.set()

    task = asyncio.create_task(await_with_cancellation(operation(), token))
    await started.wait()
    token.cancel("cancel test")

    with pytest.raises(RunCancelledError, match="cancel test"):
        await task
    assert cleaned_up.is_set()


@pytest.mark.asyncio
async def test_web_tools_reject_work_after_cancellation():
    token = CancellationToken()
    token.cancel("web cancelled")
    quick_search, deep_research = build_web_research_tools(cancellation_token=token)

    with pytest.raises(RunCancelledError, match="web cancelled"):
        await quick_search("topic")
    with pytest.raises(RunCancelledError, match="web cancelled"):
        await deep_research("topic")
