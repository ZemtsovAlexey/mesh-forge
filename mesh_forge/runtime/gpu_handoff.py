from __future__ import annotations

import logging
from typing import Any

import httpx

from mesh_forge.config import load_config

logger = logging.getLogger("mesh_forge.gpu")


def switch_vram(from_kind: str, to_kind: str) -> None:
    """Unload the previous GPU consumer before granting a different kind."""
    if from_kind == to_kind:
        return
    try:
        config = load_config()
    except Exception as exc:
        logger.warning("VRAM handoff skipped (config): %s", exc)
        return
    if not config.gpu.sequential_models:
        return
    if from_kind == "comfy":
        _free_comfyui(config.comfyui.base_url)
    elif from_kind == "llm":
        _unload_lmstudio(config.llm.base_url, config.llm.api_key)


def native_lmstudio_base(openai_base: str) -> str:
    url = (openai_base or "").rstrip("/")
    if url.endswith("/v1"):
        url = url[:-3]
    return url.rstrip("/")


def _free_comfyui(base_url: str) -> None:
    url = (base_url or "").rstrip("/")
    if not url:
        return
    try:
        with httpx.Client(timeout=15.0) as client:
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
                logger.info("ComfyUI models unloaded for GPU handoff")
    except Exception as exc:
        logger.warning("ComfyUI /free failed: %s", exc)


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
