from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from api.deps import get_orchestrator
from api.schemas import (
    GenerationSettings,
    GenerationSettingsUpdate,
    LLMModelsResponse,
    LLMSettings,
    LLMSettingsUpdate,
    SystemStatus,
)
from api.services import (
    fetch_llm_models,
    get_generation_settings,
    get_llm_settings,
    save_generation_settings,
    save_llm_settings,
    system_status,
)
router = APIRouter(prefix="/api", tags=["system"])


@router.get("/status", response_model=SystemStatus)
def get_status() -> SystemStatus:
    return system_status(get_orchestrator())


@router.get("/settings/llm", response_model=LLMSettings)
def get_llm() -> LLMSettings:
    return get_llm_settings()


@router.put("/settings/llm")
def put_llm(body: LLMSettingsUpdate) -> dict:
    if not body.planner_model or not body.vision_model:
        raise HTTPException(400, "Select planner and vision models")
    try:
        llm_status, sys_status = save_llm_settings(
            get_orchestrator(),
            base_url=body.base_url,
            api_key=body.api_key,
            planner_model=body.planner_model,
            vision_model=body.vision_model,
        )
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc
    return {"llm_status": llm_status, "system": sys_status}


@router.get("/settings/llm/models", response_model=LLMModelsResponse)
def get_llm_models(
    base_url: str = Query("http://127.0.0.1:1234/v1"),
    api_key: str = Query("lm-studio"),
) -> LLMModelsResponse:
    return fetch_llm_models(base_url, api_key)


@router.get("/settings/generation", response_model=GenerationSettings)
def get_generation() -> GenerationSettings:
    return get_generation_settings()


@router.put("/settings/generation", response_model=GenerationSettings)
async def put_generation(body: GenerationSettingsUpdate) -> GenerationSettings:
    try:
        from functools import partial

        from starlette.concurrency import run_in_threadpool

        return await run_in_threadpool(
            partial(
                save_generation_settings,
                get_orchestrator(),
                quality_preset=body.quality_preset,
                view_consistency=body.view_consistency,
                view_style=body.view_style,
                mesh_postprocess=body.mesh_postprocess,
                knobs=body.knobs.model_dump() if body.knobs is not None else None,
                download_missing=body.download_missing,
            )
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(500, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc
