from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from api.deps import get_runner
from api.schemas import (
    GenerationDefaults,
    LLMModelsResponse,
    LLMSettings,
    LLMSettingsUpdate,
    SystemStatus,
)
from mesh_forge.adapters import ComfyUiClient, LMStudioClient
from mesh_forge.config import (
    LLMConfig,
    AppConfig,
    generation_settings_payload,
    load_config,
    update_generation_settings,
    update_llm_settings,
)
from mesh_forge.runtime import get_gpu_scheduler

router = APIRouter(prefix="/api", tags=["system"])


def _system_status() -> SystemStatus:
    cfg = load_config()
    llm = LMStudioClient(cfg)
    comfy = ComfyUiClient()
    services = {"lmstudio": llm.health_check(), "comfyui": comfy.health_check()}
    snap = get_gpu_scheduler().snapshot()
    from api.schemas import GpuQueueEntry, GpuQueueInfo

    active = None
    if snap.active is not None:
        active = GpuQueueEntry(
            kind=snap.active.kind,
            label=snap.active.label,
            project_id=snap.active.project_id,
            position=snap.active.position,
        )
    waiting = [
        GpuQueueEntry(
            kind=item.kind,
            label=item.label,
            project_id=item.project_id,
            position=item.position,
        )
        for item in snap.waiting
    ]
    lines = [f"{k}: {'OK' if v else 'missing'}" for k, v in services.items()]
    lines.append(get_gpu_scheduler().status_text())
    return SystemStatus(services=services, status_text="\n".join(lines), gpu=GpuQueueInfo(active=active, waiting=waiting))


@router.get("/status", response_model=SystemStatus)
def get_status() -> SystemStatus:
    return _system_status()


@router.get("/settings/llm", response_model=LLMSettings)
def get_llm() -> LLMSettings:
    cfg = load_config()
    return LLMSettings(
        base_url=cfg.llm.base_url,
        api_key=cfg.llm.api_key,
        planner_model=cfg.llm.planner_model,
        vision_model=cfg.llm.vision_model,
    )


@router.put("/settings/llm")
def put_llm(body: LLMSettingsUpdate) -> dict:
    if not body.planner_model or not body.vision_model:
        raise HTTPException(400, "Select planner and vision models")
    try:
        update_llm_settings(
            base_url=body.base_url,
            api_key=body.api_key,
            planner_model=body.planner_model,
            vision_model=body.vision_model,
        )
        get_runner().reload_agent()
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc
    return {"ok": True, "system": _system_status().model_dump()}


@router.get("/settings/llm/models", response_model=LLMModelsResponse)
def get_llm_models(
    base_url: str = Query("http://127.0.0.1:1234/v1"),
    api_key: str = Query("lm-studio"),
) -> LLMModelsResponse:
    url = (base_url or "http://127.0.0.1:1234/v1").strip().rstrip("/")
    if not url.endswith("/v1"):
        url += "/v1"
    client = LMStudioClient(AppConfig(llm=LLMConfig(base_url=url, api_key=api_key or "lm-studio")))
    models = client.list_models()
    cfg = load_config()
    planner = cfg.llm.planner_model if cfg.llm.planner_model in models else (models[0] if models else None)
    vision = cfg.llm.vision_model if cfg.llm.vision_model in models else (models[0] if models else None)
    return LLMModelsResponse(
        models=models,
        status=client.models_status(),
        planner_model=planner,
        vision_model=vision,
    )


@router.get("/settings/generation", response_model=GenerationDefaults)
def get_generation() -> GenerationDefaults:
    payload = generation_settings_payload()
    return GenerationDefaults(
        quality_preset=str(payload.get("quality_preset") or "draft"),
        view_style=str(payload.get("view_style") or "clay"),
    )


@router.put("/settings/generation", response_model=GenerationDefaults)
def put_generation(body: GenerationDefaults) -> GenerationDefaults:
    try:
        update_generation_settings(
            quality_preset=body.quality_preset,
            view_style=body.view_style,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    payload = generation_settings_payload()
    return GenerationDefaults(
        quality_preset=str(payload.get("quality_preset") or "draft"),
        view_style=str(payload.get("view_style") or "clay"),
    )
