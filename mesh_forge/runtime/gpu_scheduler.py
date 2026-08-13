from __future__ import annotations

import itertools
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Literal

from mesh_forge import progress as prog
from mesh_forge.runtime.gpu_handoff import switch_vram

GpuKind = Literal["llm", "comfy"]
HandoffFn = Callable[[GpuKind, GpuKind], None]


@dataclass
class GpuQueueEntry:
    kind: GpuKind
    label: str
    project_id: str | None
    position: int


@dataclass
class GpuQueueSnapshot:
    active: GpuQueueEntry | None
    waiting: list[GpuQueueEntry] = field(default_factory=list)

    def status_text(self) -> str:
        if self.active is None:
            return "GPU scheduler: idle"
        wait = len(self.waiting)
        return f"GPU scheduler: busy ({self.active.label}, waiting={wait})"


@dataclass
class GpuLease:
    scheduler: "GpuScheduler"
    kind: GpuKind
    label: str
    project_id: str | None
    acquired_at: float
    token: int
    _released: bool = False

    def release(self) -> None:
        if self._released:
            return
        self.scheduler.release(self)
        self._released = True

    def __enter__(self) -> "GpuLease":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


@dataclass
class _Waiter:
    kind: GpuKind
    label: str
    project_id: str | None
    token: int


class GpuScheduler:
    def __init__(self, handoff: HandoffFn | None = None) -> None:
        self._condition = threading.Condition()
        self._waiters: deque[_Waiter] = deque()
        self._active: GpuLease | None = None
        self._last_kind: GpuKind | None = None
        self._tokens = itertools.count(1)
        self._handoff = handoff or switch_vram

    @property
    def active_project_id(self) -> str | None:
        with self._condition:
            return self._active.project_id if self._active else None

    @property
    def _active_project(self) -> str | None:
        # Back-compat for getattr(scheduler, "_active_project", None)
        return self.active_project_id

    def acquire(
        self,
        label: str,
        *,
        kind: GpuKind,
        project_id: str | None = None,
        timeout_s: int = 3600,
    ) -> GpuLease:
        started = time.time()
        waiter = _Waiter(kind=kind, label=label, project_id=project_id, token=next(self._tokens))
        prev_kind: GpuKind | None = None
        do_handoff = False
        lease: GpuLease | None = None

        with self._condition:
            self._waiters.append(waiter)

        try:
            while True:
                if project_id:
                    prog.raise_if_cancelled(project_id)
                remaining = timeout_s - (time.time() - started)
                if remaining <= 0:
                    raise TimeoutError(f"Timed out waiting for GPU lease: {label}")

                with self._condition:
                    granted = self._try_grant(waiter)
                    if granted is not None:
                        lease, prev_kind, do_handoff = granted
                        break
                    wait_stage = self._wait_stage(waiter)

                if project_id and wait_stage:
                    prog.update(project_id, 4, wait_stage)

                remaining = timeout_s - (time.time() - started)
                if remaining <= 0:
                    raise TimeoutError(f"Timed out waiting for GPU lease: {label}")
                with self._condition:
                    granted = self._try_grant(waiter)
                    if granted is not None:
                        lease, prev_kind, do_handoff = granted
                        break
                    self._condition.wait(timeout=min(0.5, remaining))
        except BaseException:
            with self._condition:
                self._drop_waiter(waiter)
                self._condition.notify_all()
            raise

        assert lease is not None
        try:
            if do_handoff and prev_kind is not None:
                try:
                    self._handoff(prev_kind, kind)
                except Exception:
                    pass
            if project_id:
                prog.raise_if_cancelled(project_id)
                prog.update(project_id, 10, f"GPU выделен: {label}")
            return lease
        except BaseException:
            lease.release()
            raise

    def _try_grant(self, waiter: _Waiter) -> tuple[GpuLease, GpuKind | None, bool] | None:
        if self._active is not None or not self._waiters or self._waiters[0] is not waiter:
            return None
        self._waiters.popleft()
        prev_kind = self._last_kind
        do_handoff = prev_kind is not None and prev_kind != waiter.kind
        lease = GpuLease(
            self,
            waiter.kind,
            waiter.label,
            waiter.project_id,
            time.time(),
            waiter.token,
        )
        self._active = lease
        self._last_kind = waiter.kind
        return lease, prev_kind, do_handoff

    def _wait_stage(self, waiter: _Waiter) -> str:
        position = 1
        for index, item in enumerate(self._waiters, start=1):
            if item is waiter:
                position = index
                break
        current = self._active.label if self._active else waiter.label
        return f"Ожидание GPU: позиция {position}, сейчас {current}"

    def release(self, lease: GpuLease) -> None:
        with self._condition:
            if self._active is not None and self._active.token == lease.token:
                self._active = None
                self._condition.notify_all()

    def wake(self) -> None:
        with self._condition:
            self._condition.notify_all()

    def holds(self, project_id: str, *, kind: GpuKind | None = None) -> bool:
        with self._condition:
            if self._active is None or self._active.project_id != project_id:
                return False
            if kind is not None and self._active.kind != kind:
                return False
            return True

    def snapshot(self) -> GpuQueueSnapshot:
        with self._condition:
            active = None
            if self._active is not None:
                active = GpuQueueEntry(
                    kind=self._active.kind,
                    label=self._active.label,
                    project_id=self._active.project_id,
                    position=0,
                )
            waiting = [
                GpuQueueEntry(
                    kind=waiter.kind,
                    label=waiter.label,
                    project_id=waiter.project_id,
                    position=index,
                )
                for index, waiter in enumerate(self._waiters, start=1)
            ]
            return GpuQueueSnapshot(active=active, waiting=waiting)

    def status_text(self) -> str:
        return self.snapshot().status_text()

    def _drop_waiter(self, waiter: _Waiter) -> None:
        try:
            self._waiters.remove(waiter)
        except ValueError:
            pass


_scheduler = GpuScheduler()


def get_gpu_scheduler() -> GpuScheduler:
    return _scheduler
