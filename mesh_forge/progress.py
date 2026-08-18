from __future__ import annotations

import contextvars
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any

_current_project_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "meshforge_gpu_project_id",
    default=None,
)


def current_project_id() -> str | None:
    return _current_project_id.get()


class OperationCancelled(Exception):
    """Raised when the user requests stop for an in-flight project job."""


@dataclass
class ProgressState:
    project_id: str
    operation: str
    percent: float
    stage: str
    active: bool
    error: str | None = None
    started_at: float = 0.0
    updated_at: float = 0.0
    cancel_requested: bool = False

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["elapsed_sec"] = round(max(0.0, time.time() - self.started_at), 1) if self.started_at else 0.0
        return data


_lock = threading.Lock()
_jobs: dict[str, ProgressState] = {}


def start(project_id: str, operation: str, stage: str = "Старт…") -> None:
    _current_project_id.set(project_id)
    now = time.time()
    with _lock:
        prev = _jobs.get(project_id)
        keep_cancel = bool(
            prev and prev.cancel_requested and (prev.active or prev.operation == "cancel")
        )
        _jobs[project_id] = ProgressState(
            project_id=project_id,
            operation=operation,
            percent=1.0,
            stage="Остановка…" if keep_cancel else stage,
            active=True,
            error=None,
            started_at=now,
            updated_at=now,
            cancel_requested=keep_cancel,
        )


def update(project_id: str, percent: float, stage: str) -> None:
    with _lock:
        job = _jobs.get(project_id)
        if not job or not job.active:
            return
        if job.cancel_requested:
            return
        job.percent = max(job.percent, min(99.0, float(percent)))
        job.stage = stage
        job.updated_at = time.time()


def finish(project_id: str, *, ok: bool = True, error: str | None = None) -> None:
    with _lock:
        job = _jobs.get(project_id)
        if not job:
            if _current_project_id.get() == project_id:
                _current_project_id.set(None)
            return
        job.active = False
        job.percent = 100.0 if ok else job.percent
        job.stage = "Готово" if ok else (error or "Ошибка")
        job.error = None if ok else (error or "Ошибка")
        job.cancel_requested = False
        job.updated_at = time.time()
    if _current_project_id.get() == project_id:
        _current_project_id.set(None)


def request_cancel(project_id: str) -> bool:
    """Mark project job as cancel-requested. Returns True if a job was active."""
    with _lock:
        job = _jobs.get(project_id)
        if not job:
            # Race: Stop arrived before this turn called start().
            _jobs[project_id] = ProgressState(
                project_id=project_id,
                operation="cancel",
                percent=0.0,
                stage="Остановка…",
                active=False,
                cancel_requested=True,
                started_at=time.time(),
                updated_at=time.time(),
            )
            return False
        if not job.active:
            return False
        job.cancel_requested = True
        job.stage = "Остановка…"
        job.updated_at = time.time()
        return bool(job.active)


def is_cancelled(project_id: str | None) -> bool:
    if not project_id:
        return False
    with _lock:
        job = _jobs.get(project_id)
        return bool(job and job.cancel_requested)


def raise_if_cancelled(project_id: str | None) -> None:
    if is_cancelled(project_id):
        raise OperationCancelled("Остановлено пользователем")


def get(project_id: str) -> ProgressState | None:
    with _lock:
        job = _jobs.get(project_id)
        return ProgressState(**asdict(job)) if job else None


def clear(project_id: str) -> None:
    with _lock:
        _jobs.pop(project_id, None)
