from __future__ import annotations

import logging
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import httpx

from mesh_forge.config import ROOT, AppConfig, llm_uses_gpu, load_config

logger = logging.getLogger("mesh_forge.gpu")

_LOOPBACK_HOSTS = frozenset(
    {"localhost", "127.0.0.1", "0.0.0.0", "::1", "[::1]", "::"}
)
_FREE_WAIT_S = 20.0
_FREE_POLL_S = 0.6
_VRAM_IDLE_BYTES = 900 * 1024 * 1024
_VRAM_DROP_BYTES = 400 * 1024 * 1024


def service_host_key(url: str) -> str:
    """Normalize a service URL to a host identity (loopback → 'local')."""
    text = (url or "").strip()
    if not text:
        return ""
    if "://" not in text:
        text = f"http://{text}"
    host = (urlparse(text).hostname or "").strip().lower().rstrip(".")
    if not host or host in _LOOPBACK_HOSTS:
        return "local"
    return host


def gpu_hosts_shared(llm_url: str, comfy_url: str) -> bool:
    """True when the LLM endpoint and ComfyUI look like the same machine."""
    llm_host = service_host_key(llm_url)
    comfy_host = service_host_key(comfy_url)
    if not llm_host or not comfy_host:
        return True
    return llm_host == comfy_host


def queues_are_split(config: AppConfig | None = None) -> bool:
    """Independent GPU queues when the two services do not share a host."""
    cfg = config or load_config()
    override = cfg.gpu.shared_gpu
    if override is True:
        return False
    if override is False:
        return True
    if not llm_uses_gpu(cfg):
        return True
    return not gpu_hosts_shared(cfg.llm.base_url, cfg.comfyui.base_url)


def switch_vram(from_kind: str | None, to_kind: str) -> None:
    """Unload the other GPU consumer before granting a different kind."""
    if from_kind == to_kind:
        return
    try:
        config = load_config()
    except Exception as exc:
        logger.warning("VRAM handoff skipped (config): %s", exc)
        return
    if not config.gpu.sequential_models:
        return
    if not llm_uses_gpu(config):
        logger.info("VRAM handoff skipped: LLM is a remote OpenAI-compatible API")
        return
    if queues_are_split(config):
        logger.info(
            "VRAM handoff skipped: LLM (%s) and ComfyUI (%s) are on different hosts",
            service_host_key(config.llm.base_url) or config.llm.base_url,
            service_host_key(config.comfyui.base_url) or config.comfyui.base_url,
        )
        return
    if to_kind == "llm":
        _free_comfyui(config.comfyui.base_url)
    elif to_kind == "comfy":
        _unload_lmstudio(config.llm.base_url, config.llm.api_key)


def native_lmstudio_base(openai_base: str) -> str:
    url = (openai_base or "").rstrip("/")
    if url.endswith("/v1"):
        url = url[:-3]
    return url.rstrip("/")


@dataclass(frozen=True)
class _VramSnap:
    vram_free: int = 0
    torch_used: int = 0

    def pretty(self) -> str:
        return (
            f"torch_used={self.torch_used / 1024**3:.2f}GiB "
            f"vram_free={self.vram_free / 1024**3:.2f}GiB"
        )


def _free_comfyui(base_url: str) -> None:
    url = (base_url or "").rstrip("/")
    if not url:
        return
    try:
        with httpx.Client(timeout=30.0) as client:
            before = _comfy_vram(client, url)
            if before is not None and before.torch_used < _VRAM_IDLE_BYTES:
                logger.info("ComfyUI already idle (%s)", before.pretty())
                _post_free(client, url)
                return
            try:
                client.post(f"{url}/interrupt")
            except Exception as exc:
                logger.debug("ComfyUI /interrupt: %s", exc)
            _post_free(client, url)
            released = _wait_comfy_vram(client, url, before)
            if released:
                return
    except Exception as exc:
        logger.warning("ComfyUI /free failed: %s", exc)
        released = False
        before = None

    if not _host_is_this_machine(url):
        logger.warning(
            "ComfyUI still holding VRAM after /free (%s); restart skipped (remote host)",
            (before.pretty() if before else "?"),
        )
        return
    logger.warning("ComfyUI /free did not drop VRAM; restarting local process")
    _restart_local_comfyui()


def _post_free(client: httpx.Client, url: str) -> None:
    response = client.post(
        f"{url}/free",
        json={"unload_models": True, "free_memory": True},
    )
    if response.status_code >= 400:
        logger.warning(
            "ComfyUI /free failed: HTTP %s %s",
            response.status_code,
            response.text[:300],
        )
    else:
        logger.info("ComfyUI /free accepted (unload_models+free_memory)")


def _wait_comfy_vram(
    client: httpx.Client,
    url: str,
    before: _VramSnap | None,
) -> bool:
    """ComfyUI /free only sets queue flags; the worker applies them up to ~1s later."""
    deadline = time.time() + _FREE_WAIT_S
    last = before
    nudged = False
    while time.time() < deadline:
        time.sleep(_FREE_POLL_S)
        last = _comfy_vram(client, url)
        if _vram_released(before, last):
            logger.info(
                "ComfyUI VRAM released (%s → %s)",
                before.pretty() if before else "?",
                last.pretty() if last else "?",
            )
            return True
        if not nudged and time.time() + _FREE_WAIT_S / 2 < deadline:
            nudged = True
            try:
                _post_free(client, url)
            except Exception:
                pass
    logger.warning(
        "ComfyUI VRAM still high after /free (%s → %s)",
        before.pretty() if before else "?",
        last.pretty() if last else "?",
    )
    return False


def _vram_released(before: _VramSnap | None, after: _VramSnap | None) -> bool:
    if after is None:
        return False
    if after.torch_used < _VRAM_IDLE_BYTES:
        return True
    if before is None:
        return False
    torch_drop = before.torch_used - after.torch_used
    freed = after.vram_free - before.vram_free
    return torch_drop >= _VRAM_DROP_BYTES or freed >= _VRAM_DROP_BYTES


def _comfy_vram(client: httpx.Client, url: str) -> _VramSnap | None:
    try:
        response = client.get(f"{url}/system_stats")
        if response.status_code >= 400:
            return None
        payload = response.json()
    except Exception:
        return None
    devices = payload.get("devices") if isinstance(payload, dict) else None
    if not isinstance(devices, list) or not devices:
        return None
    gpu = next(
        (item for item in devices if str(item.get("type", "")).lower() == "cuda"),
        devices[0],
    )
    if not isinstance(gpu, dict):
        return None
    torch_total = int(gpu.get("torch_vram_total") or 0)
    torch_free = int(gpu.get("torch_vram_free") or 0)
    vram_free = int(gpu.get("vram_free") or 0)
    torch_used = max(0, torch_total - torch_free) if torch_total else 0
    return _VramSnap(vram_free=vram_free, torch_used=torch_used)


def _host_is_this_machine(url: str) -> bool:
    if service_host_key(url) == "local":
        return True
    text = (url or "").strip()
    if "://" not in text:
        text = f"http://{text}"
    host = (urlparse(text).hostname or "").strip().lower()
    if not host:
        return False
    local_ips = {"127.0.0.1", "::1"}
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            local_ips.add(info[4][0])
    except OSError:
        pass
    return host in local_ips


def _restart_local_comfyui() -> None:
    stop = ROOT / "scripts" / "stop-comfyui.ps1"
    start = ROOT / "scripts" / "start-comfyui.ps1"
    if not stop.is_file() or not start.is_file():
        logger.warning("ComfyUI restart skipped: start/stop scripts missing")
        return
    if sys.platform != "win32":
        logger.warning("ComfyUI restart skipped: Windows scripts only")
        return
    for script, timeout in ((stop, 45), (start, 180)):
        try:
            result = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    str(script),
                ],
                cwd=str(ROOT),
                timeout=timeout,
                check=False,
                capture_output=True,
                text=True,
            )
        except Exception as exc:
            logger.warning("ComfyUI %s failed: %s", script.name, exc)
            return
        if result.returncode != 0:
            logger.warning(
                "ComfyUI %s exit %s: %s",
                script.name,
                result.returncode,
                (result.stderr or result.stdout or "")[:400],
            )
            return
    logger.info("ComfyUI restarted to release VRAM")


def _unload_lmstudio(openai_base: str, api_key: str) -> None:
    base = native_lmstudio_base(openai_base)
    if not base:
        return
    headers = {"Authorization": f"Bearer {api_key or 'lm-studio'}"}
    try:
        with httpx.Client(timeout=20.0, headers=headers) as client:
            instance_ids = _list_lmstudio_instances(client, base)
            if not instance_ids:
                logger.info("LM Studio: no loaded instances to unload")
                return
            for instance_id in instance_ids:
                _unload_one_instance(client, base, instance_id)
    except Exception as exc:
        logger.warning("LM Studio unload failed: %s", exc)


def _list_lmstudio_instances(client: httpx.Client, base: str) -> list[str]:
    for path in ("/api/v1/models", "/api/v0/models"):
        try:
            response = client.get(f"{base}{path}")
            if response.status_code >= 400:
                continue
            ids = _instance_ids_from_payload(response.json())
            if ids:
                return ids
        except Exception:
            continue
    return []


def _instance_ids_from_payload(payload: Any) -> list[str]:
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        items = payload.get("models") or payload.get("data") or payload.get("loaded") or []
    else:
        return []
    ids: list[str] = []
    for item in items:
        if isinstance(item, str) and item.strip():
            ids.append(item.strip())
            continue
        if not isinstance(item, dict):
            continue
        if item.get("loaded") is False:
            continue
        instance_id = (
            item.get("instance_id")
            or item.get("id")
            or item.get("identifier")
            or item.get("key")
        )
        if instance_id:
            ids.append(str(instance_id))
    return ids


def _unload_one_instance(client: httpx.Client, base: str, instance_id: str) -> None:
    bodies = (
        {"instance_id": instance_id},
        {"identifier": instance_id},
        {"key": instance_id},
    )
    paths = ("/api/v1/models/unload", "/api/v0/models/unload")
    for path in paths:
        for body in bodies:
            try:
                response = client.post(f"{base}{path}", json=body)
            except Exception as exc:
                logger.warning("LM Studio unload %s via %s failed: %s", instance_id, path, exc)
                continue
            if response.status_code < 400:
                logger.info("LM Studio unloaded %s via %s", instance_id, path)
                return
    logger.warning("LM Studio could not unload instance %s", instance_id)
