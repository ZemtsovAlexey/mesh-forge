from __future__ import annotations

import itertools
import logging
import threading
import time
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Literal

from mesh_forge import progress as prog
from mesh_forge.runtime.gpu_handoff import queues_are_split, service_host_key, switch_vram

logger = logging.getLogger("mesh_forge.gpu")

GpuKind = Literal["llm", "comfy"]
HandoffFn = Callable[[GpuKind | None, GpuKind], None]


def _noop_handoff(_from: GpuKind, _to: GpuKind) -> None:
    return


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
    shared: bool = True
    actives: list[GpuQueueEntry] = field(default_factory=list)
    llm_host: str = ""
    comfy_host: str = ""

    def status_text(self) -> str:
        if self.shared:
            if self.active is None:
                return "GPU scheduler: idle (shared)"
            wait = len(self.waiting)
            return f"GPU scheduler: busy ({self.active.label}, waiting={wait})"
        llm_busy = next((item.label for item in self.actives if item.kind == "llm"), None)
        comfy_busy = next((item.label for item in self.actives if item.kind == "comfy"), None)
        llm_wait = sum(1 for item in self.waiting if item.kind == "llm")
        comfy_wait = sum(1 for item in self.waiting if item.kind == "comfy")
        llm_part = llm_busy or "idle"
        if llm_wait:
            llm_part += f" wait={llm_wait}"
        comfy_part = comfy_busy or "idle"
        if comfy_wait:
            comfy_part += f" wait={comfy_wait}"
        hosts = f"{self.llm_host or '?'} != {self.comfy_host or '?'}"
        return f"GPU queues split ({hosts}): llm={llm_part}; comfy={comfy_part}"


@dataclass
class GpuLease:
    scheduler: "_GpuLane"
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


class _GpuLane:
    """FIFO lock for one GPU resource (shared device, or a single service)."""

    def __init__(self, handoff: HandoffFn | None = None, *, resource: str = "GPU") -> None:
        self._condition = threading.Condition()
        self._waiters: deque[_Waiter] = deque()
        self._active: GpuLease | None = None
        self._last_kind: GpuKind | None = None
        self._tokens = itertools.count(1)
        self._handoff = handoff or switch_vram
        self._resource = resource

    @property
    def active_project_id(self) -> str | None:
        with self._condition:
            return self._active.project_id if self._active else None

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
        waited = False

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

                waited = True
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
            if do_handoff:
                try:
                    self._handoff(prev_kind, kind)
                except Exception:
                    logger.exception("GPU VRAM handoff %s → %s failed", prev_kind, kind)
            if project_id:
                prog.raise_if_cancelled(project_id)
                if do_handoff:
                    prog.update(project_id, 10, f"GPU выделен: {label}")
                elif waited:
                    prog.update(project_id, 10, label)
            return lease
        except BaseException:
            lease.release()
            raise

    def _try_grant(self, waiter: _Waiter) -> tuple[GpuLease, GpuKind | None, bool] | None:
        if self._active is not None or not self._waiters or self._waiters[0] is not waiter:
            return None
        self._waiters.popleft()
        prev_kind = self._last_kind
        do_handoff = prev_kind != waiter.kind
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
        return f"Ожидание {self._resource}: позиция {position}, сейчас {current}"

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
            return GpuQueueSnapshot(
                active=active,
                waiting=waiting,
                actives=[active] if active else [],
            )

    def _drop_waiter(self, waiter: _Waiter) -> None:
        try:
            self._waiters.remove(waiter)
        except ValueError:
            pass


class GpuScheduler:
    """One shared GPU queue, or independent LLM/Comfy queues when hosts differ."""

    def __init__(self, handoff: HandoffFn | None = None) -> None:
        self._handoff = handoff or switch_vram
        self._shared = _GpuLane(handoff=self._handoff, resource="GPU")
        self._split: dict[GpuKind, _GpuLane] = {
            "llm": _GpuLane(handoff=_noop_handoff, resource="LM Studio"),
            "comfy": _GpuLane(handoff=_noop_handoff, resource="ComfyUI"),
        }
        self._mode_lock = threading.Lock()
        self._logged_mode: str | None = None

    def _split_queues(self) -> bool:
        try:
            from mesh_forge.config import load_config

            return queues_are_split(load_config())
        except Exception:
            return False

    def _lane(self, kind: GpuKind) -> _GpuLane:
        split = self._split_queues()
        mode = "split" if split else "shared"
        with self._mode_lock:
            if mode != self._logged_mode:
                self._log_mode(split)
                self._logged_mode = mode
        if split:
            return self._split[kind]
        return self._shared

    def _log_mode(self, split: bool) -> None:
        try:
            from mesh_forge.config import load_config

            cfg = load_config()
            llm_host = service_host_key(cfg.llm.base_url) or cfg.llm.base_url
            comfy_host = service_host_key(cfg.comfyui.base_url) or cfg.comfyui.base_url
        except Exception:
            llm_host = "?"
            comfy_host = "?"
        if split:
            logger.info(
                "GPU queues split: LLM @ %s, ComfyUI @ %s — no VRAM handoff",
                llm_host,
                comfy_host,
            )
        else:
            logger.info(
                "GPU queue shared: LLM @ %s, ComfyUI @ %s — unload on kind switch",
                llm_host,
                comfy_host,
            )

    def _all_lanes(self) -> tuple[_GpuLane, ...]:
        return (self._shared, self._split["llm"], self._split["comfy"])

    @property
    def active_project_id(self) -> str | None:
        if self._split_queues():
            return self._split["comfy"].active_project_id or self._split["llm"].active_project_id
        return self._shared.active_project_id

    @property
    def _active_project(self) -> str | None:
        return self.active_project_id

    def acquire(
        self,
        label: str,
        *,
        kind: GpuKind,
        project_id: str | None = None,
        timeout_s: int = 3600,
    ) -> GpuLease:
        return self._lane(kind).acquire(label, kind=kind, project_id=project_id, timeout_s=timeout_s)

    def wake(self) -> None:
        for lane in self._all_lanes():
            lane.wake()

    def holds(self, project_id: str, *, kind: GpuKind | None = None) -> bool:
        if kind is not None:
            if self._lane(kind).holds(project_id, kind=kind):
                return True
            return any(lane.holds(project_id, kind=kind) for lane in self._all_lanes())
        return any(lane.holds(project_id) for lane in self._all_lanes())

    def snapshot(self) -> GpuQueueSnapshot:
        try:
            from mesh_forge.config import load_config

            cfg = load_config()
            llm_host = service_host_key(cfg.llm.base_url)
            comfy_host = service_host_key(cfg.comfyui.base_url)
            split = queues_are_split(cfg)
        except Exception:
            llm_host = ""
            comfy_host = ""
            split = False
        if not split:
            snap = self._shared.snapshot()
            snap.shared = True
            snap.actives = [snap.active] if snap.active else []
            snap.llm_host = llm_host
            snap.comfy_host = comfy_host
            return snap
        llm = self._split["llm"].snapshot()
        comfy = self._split["comfy"].snapshot()
        actives = [item for item in (llm.active, comfy.active) if item is not None]
        return GpuQueueSnapshot(
            active=actives[0] if actives else None,
            waiting=list(llm.waiting) + list(comfy.waiting),
            shared=False,
            actives=actives,
            llm_host=llm_host,
            comfy_host=comfy_host,
        )

    def status_text(self) -> str:
        return self.snapshot().status_text()


_scheduler = GpuScheduler()


def get_gpu_scheduler() -> GpuScheduler:
    return _scheduler


@contextmanager
def acquire_llm(*, project_id: str | None = None, label: str | None = None) -> Iterator[GpuLease | None]:
    """Take the LLM GPU slot for local LM Studio; no-op for remote OpenAI APIs."""
    from mesh_forge.config import llm_display_name, llm_uses_gpu, load_config

    try:
        cfg = load_config()
        uses_gpu = llm_uses_gpu(cfg)
        slot_label = label or llm_display_name(cfg)
    except Exception:
        uses_gpu = True
        slot_label = label or "LM Studio"
    if not uses_gpu:
        yield None
        return
    with get_gpu_scheduler().acquire(slot_label, kind="llm", project_id=project_id) as lease:
        yield lease
