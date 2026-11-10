"""In-process worker supervision for durable local research runs."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from contextlib import suppress
from datetime import UTC, datetime, timedelta
import logging
from typing import Any
from uuid import uuid4

from backend.config import Settings
from backend.events import RunEventType
from backend.research.run_manager import resume_research, run_research
from backend.runtime.cancellation import CancellationToken
from backend.runtime.run_journal import RunJournal
from backend.runtime.run_state import RunStatus, transition_run_record
from backend.storage.run_paths import generate_run_id, load_run_paths
from backend.storage.sqlite_runtime import SQLiteRuntimeStore


logger = logging.getLogger(__name__)


def _consume_background_result(task: asyncio.Task[object]) -> None:
    """Retrieve persisted task failures so asyncio does not report orphan errors."""

    with suppress(BaseException):
        task.result()


class RunSupervisor:
    """Own local async tasks while SQLite remains the durable source of truth."""

    def __init__(
        self,
        settings: Settings,
        *,
        store: SQLiteRuntimeStore | None = None,
        worker_id: str | None = None,
        heartbeat_interval_seconds: float = 5.0,
        stale_after_seconds: float = 30.0,
        watchdog_interval_seconds: float = 5.0,
    ) -> None:
        if heartbeat_interval_seconds <= 0:
            raise ValueError("heartbeat interval must be positive")
        if stale_after_seconds < 0:
            raise ValueError("stale threshold must be non-negative")
        if watchdog_interval_seconds <= 0:
            raise ValueError("watchdog interval must be positive")
        self.settings = settings
        self.store = store or SQLiteRuntimeStore(
            settings.resolved_runtime_database_path()
        )
        self.worker_id = worker_id or f"local-{uuid4().hex}"
        self.heartbeat_interval_seconds = heartbeat_interval_seconds
        self.stale_after_seconds = stale_after_seconds
        self.watchdog_interval_seconds = watchdog_interval_seconds
        self._tasks: dict[str, asyncio.Task[object]] = {}
        self._tokens: dict[str, CancellationToken] = {}
        self._lock = asyncio.Lock()
        self._watchdog_task: asyncio.Task[None] | None = None

    async def start(self) -> tuple[str, ...]:
        """Recover abandoned leases once, then start periodic supervision."""

        recovered = await self.recover_stale_once()
        if self._watchdog_task is None or self._watchdog_task.done():
            self._watchdog_task = asyncio.create_task(
                self._watchdog_loop(), name="research-watchdog"
            )
        return recovered

    async def shutdown(self) -> None:
        """Interrupt active work so run checkpoints remain resumable."""

        watchdog = self._watchdog_task
        self._watchdog_task = None
        if watchdog is not None:
            watchdog.cancel()
            with suppress(asyncio.CancelledError):
                await watchdog
        async with self._lock:
            tasks = tuple(self._tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def start_run(
        self,
        query: str,
        source_mode: str,
        *,
        guidelines: list[str] | None = None,
        run_id: str | None = None,
    ) -> str:
        selected_run_id = run_id or generate_run_id()
        if self.store.get_run(selected_run_id) is not None:
            raise ValueError(f"run already exists: {selected_run_id}")
        token = CancellationToken()
        started = asyncio.Event()
        operation = run_research(
            query,
            source_mode,
            self.settings,
            guidelines=guidelines,
            run_id=selected_run_id,
            cancellation_token=token,
            started_event=started,
        )
        await self._launch(selected_run_id, token, started, operation)
        return selected_run_id

    async def resume_run(self, run_id: str) -> str:
        record = self.store.get_run(run_id)
        if record is None:
            raise LookupError(f"unknown run: {run_id}")
        if record.status not in {RunStatus.RUNNING, RunStatus.INTERRUPTED}:
            raise ValueError(
                f"run {run_id} cannot resume from status {record.status.value}"
            )
        token = CancellationToken()
        started = asyncio.Event()
        operation = resume_research(
            run_id,
            self.settings,
            cancellation_token=token,
            started_event=started,
        )
        await self._launch(run_id, token, started, operation)
        return run_id

    async def cancel_run(
        self, run_id: str, reason: str = "Run cancelled by API request."
    ) -> bool:
        async with self._lock:
            token = self._tokens.get(run_id)
        if token is None:
            return False
        return token.cancel(reason)

    async def is_active(self, run_id: str) -> bool:
        async with self._lock:
            task = self._tasks.get(run_id)
            return task is not None and not task.done()

    async def active_run_ids(self) -> tuple[str, ...]:
        async with self._lock:
            return tuple(
                run_id for run_id, task in self._tasks.items() if not task.done()
            )

    async def recover_stale_once(self) -> tuple[str, ...]:
        """Mark leases with expired heartbeats as resumable interruptions."""

        before = datetime.now(UTC) - timedelta(seconds=self.stale_after_seconds)
        leases = await asyncio.to_thread(self.store.stale_worker_leases, before=before)
        recovered: list[str] = []
        for lease in leases:
            if await self.is_active(lease.run_id):
                continue
            record = await asyncio.to_thread(self.store.get_run, lease.run_id)
            if record is not None and record.status is RunStatus.RUNNING:
                reason = (
                    "Worker heartbeat expired; the run can be resumed from its "
                    "latest durable checkpoint."
                )
                payload = {
                    "reason": "worker_heartbeat_expired",
                    "worker_id": lease.worker_id,
                }
                try:
                    try:
                        paths = load_run_paths(
                            self.settings.resolved_outputs_dir(), lease.run_id
                        )
                    except FileNotFoundError:
                        interrupted = transition_run_record(
                            record,
                            status=RunStatus.INTERRUPTED,
                            stage=record.stage,
                            finished_at=datetime.now(UTC),
                            error=reason,
                        )
                        await asyncio.to_thread(
                            self.store.commit,
                            interrupted,
                            RunEventType.RUN_INTERRUPTED,
                            source="watchdog",
                            payload=payload,
                        )
                    else:
                        journal = RunJournal.open(
                            self.settings.resolved_runtime_database_path(), paths
                        )
                        journal.transition(
                            RunEventType.RUN_INTERRUPTED,
                            status=RunStatus.INTERRUPTED,
                            stage=journal.record.stage,
                            finished_at=datetime.now(UTC),
                            error=reason,
                            payload=payload,
                        )
                    recovered.append(lease.run_id)
                finally:
                    await asyncio.to_thread(
                        self.store.finish_worker,
                        lease.run_id,
                        lease.worker_id,
                        state="stale",
                    )
            else:
                await asyncio.to_thread(
                    self.store.finish_worker,
                    lease.run_id,
                    lease.worker_id,
                    state="stale",
                )
        return tuple(recovered)

    async def _launch(
        self,
        run_id: str,
        token: CancellationToken,
        started: asyncio.Event,
        operation: Coroutine[Any, Any, object],
    ) -> None:
        async with self._lock:
            existing = self._tasks.get(run_id)
            if existing is not None and not existing.done():
                if hasattr(operation, "close"):
                    operation.close()  # type: ignore[attr-defined]
                raise ValueError(f"run is already active: {run_id}")
            managed = asyncio.create_task(
                self._manage_run(run_id, started, operation),
                name=f"research-run-{run_id}",
            )
            managed.add_done_callback(_consume_background_result)
            self._tasks[run_id] = managed
            self._tokens[run_id] = token
        await self._wait_until_started(started, managed)

    async def _wait_until_started(
        self, started: asyncio.Event, managed: asyncio.Task[object]
    ) -> None:
        waiter = asyncio.create_task(started.wait())
        try:
            done, _ = await asyncio.wait(
                {waiter, managed}, return_when=asyncio.FIRST_COMPLETED
            )
            if managed in done:
                await managed
        finally:
            waiter.cancel()
            with suppress(asyncio.CancelledError):
                await waiter

    async def _manage_run(
        self,
        run_id: str,
        started: asyncio.Event,
        operation: Coroutine[Any, Any, object],
    ) -> object:
        runner = asyncio.create_task(operation)  # type: ignore[arg-type]
        starter = asyncio.create_task(started.wait())
        heartbeat: asyncio.Task[None] | None = None
        claimed = False
        try:
            done, _ = await asyncio.wait(
                {runner, starter}, return_when=asyncio.FIRST_COMPLETED
            )
            if runner in done:
                starter.cancel()
                with suppress(asyncio.CancelledError):
                    await starter
                return await runner
            await asyncio.to_thread(self.store.claim_worker, run_id, self.worker_id)
            claimed = True
            heartbeat = asyncio.create_task(
                self._heartbeat_loop(run_id), name=f"heartbeat-{run_id}"
            )
            done, _ = await asyncio.wait(
                {runner, heartbeat}, return_when=asyncio.FIRST_COMPLETED
            )
            if heartbeat in done:
                runner.cancel()
                with suppress(asyncio.CancelledError):
                    await runner
                await heartbeat
            return await runner
        finally:
            starter.cancel()
            with suppress(asyncio.CancelledError):
                await starter
            runner.cancel()
            with suppress(asyncio.CancelledError):
                await runner
            if heartbeat is not None:
                heartbeat.cancel()
                with suppress(asyncio.CancelledError):
                    await heartbeat
            if claimed:
                await asyncio.to_thread(
                    self.store.finish_worker,
                    run_id,
                    self.worker_id,
                    state="finished",
                )
            async with self._lock:
                self._tasks.pop(run_id, None)
                self._tokens.pop(run_id, None)

    async def _heartbeat_loop(self, run_id: str) -> None:
        while True:
            await asyncio.sleep(self.heartbeat_interval_seconds)
            await asyncio.to_thread(
                self.store.heartbeat_worker, run_id, self.worker_id
            )

    async def _watchdog_loop(self) -> None:
        while True:
            await asyncio.sleep(self.watchdog_interval_seconds)
            try:
                await self.recover_stale_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Watchdog pass failed; supervision will continue")


__all__ = ["RunSupervisor"]
