"""Run-scoped cooperative cancellation shared across async and thread code."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from contextlib import suppress
import threading
from typing import TypeVar


_T = TypeVar("_T")


class RunCancelledError(asyncio.CancelledError):
    """Raised when a user-requested run cancellation reaches a checkpoint."""


def _resolve_waiter(waiter: asyncio.Future[None]) -> None:
    if not waiter.done():
        waiter.set_result(None)


class CancellationToken:
    """Thread-safe, idempotent cancellation signal for one research run."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._reason = "Run cancelled by request."
        self._waiters: set[tuple[asyncio.AbstractEventLoop, asyncio.Future[None]]] = (
            set()
        )

    @property
    def is_cancelled(self) -> bool:
        return self._event.is_set()

    @property
    def reason(self) -> str:
        with self._lock:
            return self._reason

    def cancel(self, reason: str = "Run cancelled by request.") -> bool:
        """Signal cancellation once; return whether this call changed state."""

        normalized = reason.strip() or "Run cancelled by request."
        with self._lock:
            if self._event.is_set():
                return False
            self._reason = normalized
            self._event.set()
            waiters = tuple(self._waiters)
        for loop, waiter in waiters:
            loop.call_soon_threadsafe(_resolve_waiter, waiter)
        return True

    def raise_if_cancelled(self) -> None:
        if self.is_cancelled:
            raise RunCancelledError(self.reason)

    def sleep(self, delay_seconds: float) -> None:
        """Block for a retry delay, waking immediately when cancelled."""

        if delay_seconds < 0:
            raise ValueError("delay_seconds must be non-negative")
        if self._event.wait(delay_seconds):
            raise RunCancelledError(self.reason)

    async def wait(self) -> None:
        """Wait without polling or occupying an executor thread."""

        if self.is_cancelled:
            return
        loop = asyncio.get_running_loop()
        waiter: asyncio.Future[None] = loop.create_future()
        key = (loop, waiter)
        with self._lock:
            if self._event.is_set():
                waiter.set_result(None)
            else:
                self._waiters.add(key)
        try:
            await waiter
        finally:
            with self._lock:
                self._waiters.discard(key)


async def await_with_cancellation(
    awaitable: Awaitable[_T], cancellation_token: CancellationToken
) -> _T:
    """Cancel an async operation when the run token is signalled."""

    cancellation_token.raise_if_cancelled()
    operation = asyncio.ensure_future(awaitable)
    cancellation_waiter = asyncio.create_task(cancellation_token.wait())
    try:
        done, _ = await asyncio.wait(
            {operation, cancellation_waiter},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if operation in done:
            cancellation_waiter.cancel()
            with suppress(asyncio.CancelledError):
                await cancellation_waiter
            return await operation

        operation.cancel()
        with suppress(asyncio.CancelledError):
            await operation
        raise RunCancelledError(cancellation_token.reason)
    except BaseException:
        operation.cancel()
        cancellation_waiter.cancel()
        with suppress(asyncio.CancelledError):
            await operation
        with suppress(asyncio.CancelledError):
            await cancellation_waiter
        raise


__all__ = [
    "CancellationToken",
    "RunCancelledError",
    "await_with_cancellation",
]
