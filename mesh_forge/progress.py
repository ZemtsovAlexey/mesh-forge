from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass
from typing import Any


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

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["elapsed_sec"] = round(max(0.0, time.time() - self.started_at), 1) if self.started_at else 0.0
        return data


_lock = threading.Lock()
_jobs: dict[str, ProgressState] = {}


def start(project_id: str, operation: str, stage: str = "Старт…") -> None:
    now = time.time()
    with _lock:
        _jobs[project_id] = ProgressState(
            project_id=project_id,
            operation=operation,
            percent=1.0,
            stage=stage,
            active=True,
            error=None,
            started_at=now,
            updated_at=now,
        )


def update(project_id: str, percent: float, stage: str) -> None:
    with _lock:
        job = _jobs.get(project_id)
        if not job or not job.active:
            return
        job.percent = max(job.percent, min(99.0, float(percent)))
        job.stage = stage
        job.updated_at = time.time()


def finish(project_id: str, *, ok: bool = True, error: str | None = None) -> None:
    with _lock:
        job = _jobs.get(project_id)
        if not job:
            return
        job.active = False
        job.percent = 100.0 if ok else job.percent
        job.stage = "Готово" if ok else (error or "Ошибка")
        job.error = None if ok else (error or "Ошибка")
        job.updated_at = time.time()


def get(project_id: str) -> ProgressState | None:
    with _lock:
        job = _jobs.get(project_id)
        return ProgressState(**asdict(job)) if job else None


def clear(project_id: str) -> None:
    with _lock:
        _jobs.pop(project_id, None)
