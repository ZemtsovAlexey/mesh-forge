from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from mesh_forge import progress as prog


@dataclass
class GpuLease:
    scheduler: "GpuScheduler"
    label: str
    project_id: str | None
    acquired_at: float
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


class GpuScheduler:
    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._active_label: str | None = None
        self._active_project: str | None = None
        self._waiting = 0

    def acquire(
        self,
        label: str,
        *,
        project_id: str | None = None,
        timeout_s: int = 3600,
    ) -> GpuLease:
        started = time.time()
        with self._condition:
            self._waiting += 1
            try:
                if project_id:
                    prog.update(project_id, 4, f"Ожидание GPU: {label}…")
                while self._active_label is not None:
                    remaining = timeout_s - (time.time() - started)
                    if remaining <= 0:
                        raise TimeoutError(f"Timed out waiting for GPU lease: {label}")
                    self._condition.wait(timeout=min(0.5, remaining))
                self._active_label = label
                self._active_project = project_id
            finally:
                self._waiting -= 1

        if project_id:
            prog.update(project_id, 10, f"GPU выделен: {label}")
        return GpuLease(self, label, project_id, time.time())

    def release(self, lease: GpuLease) -> None:
        with self._condition:
            if self._active_label == lease.label and self._active_project == lease.project_id:
                self._active_label = None
                self._active_project = None
                self._condition.notify_all()

    def status_text(self) -> str:
        with self._condition:
            if self._active_label is None:
                return "GPU scheduler: idle"
            return f"GPU scheduler: busy ({self._active_label}, waiting={self._waiting})"


_scheduler = GpuScheduler()


def get_gpu_scheduler() -> GpuScheduler:
    return _scheduler
