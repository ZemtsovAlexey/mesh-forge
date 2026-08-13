from __future__ import annotations

import logging
import shutil
import tempfile
import traceback
from pathlib import Path

from mesh_forge.config import (
    AppConfig,
    LLMConfig,
    download_comfyui_checkpoints,
    generation_settings_payload,
    load_config,
    missing_comfyui_checkpoints,
    update_generation_settings,
    update_llm_settings,
)
from mesh_forge.domain import GenerationJob, JobOptions
from mesh_forge.manifest import ProjectManifest, list_projects
from mesh_forge.mesh_qc import analyze_mesh, is_print_ready
from mesh_forge.orchestrator import Orchestrator
from mesh_forge import progress as prog
from mesh_forge.adapters import LMStudioClient

from api.schemas import (
    ArtifactInfo,
    ExportInfo,
    GenerationSettings,
    LLMModelsResponse,
    LLMSettings,
    OperationResult,
    PipelineStateInfo,
    ProjectDetail,
    ProjectSummary,
    SystemStatus,
    VersionInfo,
)


logger = logging.getLogger("api.services")


def _mesh_url(project_id: str, manifest: ProjectManifest) -> str | None:
    if manifest.current_mesh_path():
        return f"/api/projects/{project_id}/mesh"
    return None


def project_summary(manifest: ProjectManifest) -> ProjectSummary:
    return ProjectSummary(
        id=manifest.id,
        name=manifest.name,
        current_version=manifest.current_version,
        has_mesh=manifest.current_mesh_path() is not None,
    )


def project_detail(manifest: ProjectManifest) -> ProjectDetail:
    versions = [
        VersionInfo(
            version=v.version,
            branch=v.branch,
            action=v.action,
            instruction=v.instruction,
            qc=v.qc,
            artifacts=[
                ArtifactInfo(
                    kind=artifact.kind,
                    path=artifact.path,
                    label=artifact.label,
                    stage=artifact.stage,
                    source=artifact.source,
                )
                for artifact in v.artifacts
            ],
            created_at=v.created_at,
        )
        for v in manifest.versions
    ]
    return ProjectDetail(
        id=manifest.id,
        name=manifest.name,
        current_version=manifest.current_version,
        has_mesh=manifest.current_mesh_path() is not None,
        mesh_url=_mesh_url(manifest.id, manifest),
        versions=versions,
    )


def list_project_summaries() -> list[ProjectSummary]:
    return [project_summary(p) for p in list_projects()]


def qc_report_for(manifest: ProjectManifest) -> str | None:
    mesh = manifest.current_mesh_path()
    if not mesh:
        return None
    return analyze_mesh(mesh).summary()


def operation_result(
    manifest: ProjectManifest,
    message: str,
    *,
    pipeline: dict | None = None,
) -> OperationResult:
    pipe = None
    if pipeline is not None:
        pipe = PipelineStateInfo(**pipeline)
    return OperationResult(
        message=message,
        project=project_detail(manifest),
        qc_report=qc_report_for(manifest),
        pipeline=pipe,
    )


def pipeline_result(manifest: ProjectManifest, state_payload: dict, message: str | None = None) -> OperationResult:
    return operation_result(
        manifest,
        message or state_payload.get("message") or "OK",
        pipeline=state_payload,
    )


def run_safe(orch: Orchestrator, fn, *, operation: str, project_id: str) -> OperationResult:
    logger.info("[%s] project=%s start", operation, project_id)
    prog.start(project_id, operation, stage="Подготовка…")
    try:
        manifest, message = fn()
        prog.finish(project_id, ok=True)
        logger.info("[%s] project=%s done: %s", operation, project_id, message)
        return operation_result(manifest, message)
    except Exception as exc:
        prog.finish(project_id, ok=False, error=str(exc)[:200])
        logger.error(
            "[%s] project=%s failed: %s\n%s",
            operation,
            project_id,
            exc,
            traceback.format_exc(),
        )
        detail = traceback.format_exc()[-1200:]
        raise RuntimeError(f"{exc}\n{detail}") from exc


def run_generation_job(
    orch: Orchestrator,
    manifest: ProjectManifest,
    job: GenerationJob,
) -> OperationResult:
    def _run():
        return orch.run_job(manifest, job)

    operation = "job"
    if job.has_mesh():
        operation = "edit" if (job.has_prompt() or job.has_images()) else "repair"
    elif job.has_prompt() and not job.has_images():
        operation = "text_to_mesh"
    elif job.has_images():
        operation = "image_to_3d"
    return run_safe(orch, _run, operation=operation, project_id=manifest.id)


def export_info(manifest: ProjectManifest) -> ExportInfo:
    mesh = manifest.current_mesh_path()
    if not mesh:
        return ExportInfo(report="No mesh to export", print_ready=False)
    stats = analyze_mesh(mesh)
    ready = is_print_ready(stats)
    report = stats.summary() + f"\n\nPrint ready: {'YES' if ready else 'NO — fix before slicing'}"
    return ExportInfo(
        report=report,
        print_ready=ready,
        download_url=f"/api/projects/{manifest.id}/mesh",
    )


def system_status(orch: Orchestrator) -> SystemStatus:
    services = orch.system_status()
    return SystemStatus(services=services, status_text=orch.status_text())


def get_llm_settings() -> LLMSettings:
    cfg = load_config()
    return LLMSettings(
        base_url=cfg.llm.base_url,
        api_key=cfg.llm.api_key,
        planner_model=cfg.llm.planner_model,
        vision_model=cfg.llm.vision_model,
    )


def llm_client_for(base_url: str, api_key: str) -> LMStudioClient:
    url = (base_url or "http://127.0.0.1:1234/v1").strip().rstrip("/")
    if not url.endswith("/v1"):
        url += "/v1"
    tmp = AppConfig(llm=LLMConfig(base_url=url, api_key=api_key or "lm-studio"))
    return LMStudioClient(tmp)


def fetch_llm_models(base_url: str, api_key: str) -> LLMModelsResponse:
    client = llm_client_for(base_url, api_key)
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


def save_llm_settings(
    orch: Orchestrator,
    *,
    base_url: str,
    api_key: str,
    planner_model: str,
    vision_model: str,
) -> tuple[str, SystemStatus]:
    update_llm_settings(
        base_url=base_url,
        api_key=api_key or "lm-studio",
        planner_model=planner_model,
        vision_model=vision_model,
    )
    orch.reload_config()
    return orch.llm.models_status(), system_status(orch)


def get_generation_settings() -> GenerationSettings:
    return GenerationSettings(**generation_settings_payload())


def save_generation_settings(
    orch: Orchestrator,
    *,
    quality_preset: str,
    view_consistency: str = "img2img",
    view_style: str = "clay",
    mesh_postprocess: bool = True,
    knobs: dict | None = None,
    download_missing: bool = True,
) -> GenerationSettings:
    cfg = update_generation_settings(
        quality_preset=quality_preset,
        view_consistency=view_consistency,
        view_style=view_style,
        mesh_postprocess=mesh_postprocess,
        knobs=knobs,
    )
    report = None
    if download_missing:
        missing = missing_comfyui_checkpoints(cfg)
        if missing:
            report = download_comfyui_checkpoints(missing, config=cfg)
            if report.get("errors"):
                # Keep preset saved, but surface download failures.
                still = missing_comfyui_checkpoints(cfg)
                if still:
                    raise RuntimeError(
                        "Preset сохранён, но не удалось скачать checkpoints: "
                        + "; ".join(report["errors"])
                    )
    orch.reload_config()
    return GenerationSettings(**generation_settings_payload(cfg, download_report=report))


def save_upload_to_tmp(upload_file, suffix: str) -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="meshforge_"))
    dest = tmp / f"upload{suffix}"
    with dest.open("wb") as out:
        shutil.copyfileobj(upload_file.file, out)
    logger.debug("Saved upload to %s (%d bytes)", dest, dest.stat().st_size)
    return dest


def build_job(
    *,
    project_id: str,
    prompt: str = "",
    image_paths: list[Path] | None = None,
    source_mesh: Path | None = None,
    use_current_mesh: bool = False,
    backend: str = "auto",
    remove_bg: bool = True,
    solidify_mm: float = 0.0,
    scan_mode: str = "light",
    smooth_iters: int = 1,
    view_count: int = 4,
    user_prompt: str = "",
    generation_prompt: str = "",
    semantic_regen: bool = False,
    guided_edit: bool = False,
    planned_ops: list[dict] | None = None,
    anchor_image: Path | None = None,
) -> GenerationJob:
    generation = (generation_prompt or prompt or "").strip()
    return GenerationJob(
        project_id=project_id,
        prompt=generation,
        image_paths=image_paths or [],
        source_mesh=source_mesh,
        options=JobOptions(
            backend=backend or "auto",
            remove_bg=remove_bg,
            solidify_mm=solidify_mm,
            scan_mode=scan_mode,
            smooth_iters=smooth_iters,
            use_current_mesh=use_current_mesh,
            view_count=view_count,
            user_prompt=(user_prompt or "").strip(),
            generation_prompt=generation,
            semantic_regen=semantic_regen,
            guided_edit=guided_edit,
            planned_ops=list(planned_ops or []),
            anchor_image=anchor_image,
        ),
    )


def chat_state_response(data: dict) -> dict:
    return data


def get_project_chat(manifest: ProjectManifest) -> dict:
    from mesh_forge.application import PromptChatService
    from mesh_forge.application.chat_results import enrich_chat_payload
    from mesh_forge.application.notebook import notebook_payload
    from mesh_forge.application.stepped_pipeline import pipeline_payload

    data = PromptChatService().get(manifest)
    data["pipeline"] = pipeline_payload(manifest)
    data["notebook"] = notebook_payload(manifest)
    return enrich_chat_payload(manifest, data)


def post_project_chat_message(
    manifest: ProjectManifest,
    *,
    text: str,
    has_images: bool = False,
    ref_ids: list[str] | None = None,
) -> dict:
    from mesh_forge.application.chat_agent import ChatAgentService

    return ChatAgentService().post_message(
        manifest,
        text,
        has_images=has_images,
        ref_ids=ref_ids,
    )


def restart_project_chat_from_message(manifest: ProjectManifest, message_id: str) -> dict:
    from mesh_forge.application.chat_agent import ChatAgentService

    return ChatAgentService().restart_from_message(manifest, message_id)


def confirm_project_chat(
    orch: Orchestrator,
    manifest: ProjectManifest,
    *,
    image_paths: list[Path] | None = None,
    solidify_mm: float = 0.0,
    mode: str = "light",
    smooth_iters: int = 1,
    remove_bg: bool = True,
) -> OperationResult:
    from mesh_forge.application import PromptChatService
    from mesh_forge.application.stepped_pipeline import (
        pipeline_payload,
        start_photo_gate,
        start_text_front,
    )

    chat = PromptChatService()
    state = chat.load(manifest)
    image_paths = image_paths or []

    # Manual cleanup path is separate; chat confirm is create / semantic regen / geometry / images.
    if state.intent == "geometry_edit":
        if not state.ready:
            raise ValueError("Chat is not ready for geometry edit")
        current = manifest.current_mesh_path()
        if current is None or not current.is_file():
            raise ValueError("No current mesh to edit")
        instruction = (state.user_prompt or state.edit_brief_en or "cleanup mesh").strip()
        job = build_job(
            project_id=manifest.id,
            prompt=instruction,
            user_prompt=instruction,
            generation_prompt=instruction,
            source_mesh=current,
            use_current_mesh=True,
            semantic_regen=False,
            planned_ops=list(state.planned_ops or []),
            solidify_mm=solidify_mm,
        )
        result = run_generation_job(orch, manifest, job)
        state = chat.load(manifest)
        state.ready = False
        state.status = "idle"
        state.draft_prompt_en = ""
        state.edit_brief_en = ""
        chat.save(manifest, state)
        return result

    if state.intent == "guided_edit":
        if not state.ready or not (state.edit_brief_en or state.draft_prompt_en).strip():
            raise ValueError("Chat is not ready for guided edit")
        brief = (state.edit_brief_en or state.draft_prompt_en).strip()
        current = manifest.current_mesh_path()
        # Do not pass photo refs as img2img anchors; runner bakes clay front when needed.
        anchor = manifest.find_view_anchor()
        job = build_job(
            project_id=manifest.id,
            prompt=brief,
            user_prompt=state.user_prompt or brief,
            generation_prompt=brief,
            source_mesh=current,
            use_current_mesh=bool(current),
            guided_edit=True,
            semantic_regen=False,
            anchor_image=anchor,
            solidify_mm=solidify_mm,
        )
        result = run_generation_job(orch, manifest, job)
        state = chat.load(manifest)
        state.ready = False
        state.status = "idle"
        state.draft_prompt_en = ""
        state.edit_brief_en = ""
        chat.save(manifest, state)
        return result

    if state.intent == "semantic_edit" or (
        state.mode == "edit"
        and state.edit_brief_en
        and state.intent not in {"geometry_edit", "guided_edit", "create"}
    ):
        if not state.ready or not (state.edit_brief_en or state.draft_prompt_en).strip():
            raise ValueError("Chat is not ready for semantic regenerate")
        brief = (state.edit_brief_en or state.draft_prompt_en).strip()
        job = build_job(
            project_id=manifest.id,
            prompt=brief,
            user_prompt=state.user_prompt or brief,
            generation_prompt=brief,
            semantic_regen=True,
            solidify_mm=solidify_mm,
        )
        result = run_generation_job(orch, manifest, job)
        # Keep history; just clear ready flags
        state = chat.load(manifest)
        state.ready = False
        state.status = "idle"
        state.draft_prompt_en = ""
        state.edit_brief_en = ""
        chat.save(manifest, state)
        return result

    # Text create: always stepped front-only (even if project already has a mesh).
    # Full semantic/guided paths are handled above by intent.
    if state.intent == "create" or (state.draft_prompt_en.strip() and not (state.edit_brief_en or "").strip()):
        if not state.ready or not state.draft_prompt_en.strip():
            raise ValueError("Chat is not ready to generate")
        brief = state.draft_prompt_en.strip()
        user_prompt = state.user_prompt or brief
        pipe = start_text_front(
            manifest,
            brief_en=brief,
            user_prompt=user_prompt,
            solidify_mm=solidify_mm,
        )
        state = chat.load(manifest)
        state.ready = False
        state.status = "pipeline"
        # Keep the EN brief that was actually sent to Comfy (may have been translated).
        if pipe.brief_en:
            state.draft_prompt_en = pipe.brief_en
        chat.save(manifest, state)
        return pipeline_result(manifest, pipeline_payload(manifest, pipe))

    # Photo gate: preview before mesh
    if image_paths and not state.draft_prompt_en:
        pipe = start_photo_gate(
            manifest,
            image_paths,
            user_prompt=state.user_prompt or "",
            solidify_mm=solidify_mm,
            remove_bg=remove_bg,
        )
        state = chat.load(manifest)
        state.ready = False
        state.status = "pipeline"
        chat.save(manifest, state)
        return pipeline_result(manifest, pipeline_payload(manifest, pipe))

    raise ValueError("Chat is not ready to generate")


def get_pipeline(manifest: ProjectManifest) -> dict:
    from mesh_forge.application.stepped_pipeline import pipeline_payload

    return pipeline_payload(manifest)


def continue_project_pipeline(manifest: ProjectManifest) -> OperationResult:
    from mesh_forge.application.stepped_pipeline import continue_pipeline, pipeline_payload

    pipe = continue_pipeline(manifest)
    return pipeline_result(manifest, pipeline_payload(manifest, pipe))


def redo_project_pipeline(
    manifest: ProjectManifest,
    *,
    step: str = "front",
    brief_en: str | None = None,
) -> OperationResult:
    from mesh_forge.application.stepped_pipeline import pipeline_payload, redo_step

    pipe = redo_step(manifest, step=step, brief_en=brief_en)
    return pipeline_result(manifest, pipeline_payload(manifest, pipe))


def _extract_generation_prompt(instruction: str | None) -> tuple[str, str]:
    """Return (generation_prompt, user_prompt) from a version instruction."""
    text = (instruction or "").strip()
    if not text:
        return "", ""
    user_prompt = ""
    generation = ""
    lower = text.lower()
    if "generation:" in lower:
        # Split user: / generation: blocks
        lines = text.splitlines()
        buf_user: list[str] = []
        buf_gen: list[str] = []
        mode = ""
        for line in lines:
            low = line.strip().lower()
            if low.startswith("user:"):
                mode = "user"
                rest = line.split(":", 1)[1].strip() if ":" in line else ""
                if rest:
                    buf_user.append(rest)
                continue
            if low.startswith("generation:"):
                mode = "gen"
                rest = line.split(":", 1)[1].strip() if ":" in line else ""
                if rest:
                    buf_gen.append(rest)
                continue
            if mode == "user":
                buf_user.append(line)
            elif mode == "gen":
                buf_gen.append(line)
        user_prompt = "\n".join(buf_user).strip()
        generation = "\n".join(buf_gen).strip()
    if not generation:
        generation = text
    if not user_prompt:
        user_prompt = generation
    return generation, user_prompt


def regenerate_project(
    orch: Orchestrator,
    manifest: ProjectManifest,
    *,
    solidify_mm: float = 0.0,
) -> OperationResult:
    """Re-run text→mesh using the last stored generation prompt."""
    for entry in reversed(manifest.versions):
        generation, user_prompt = _extract_generation_prompt(entry.instruction)
        if not generation.strip():
            continue
        semantic = (entry.branch or "").startswith("regen") or (entry.action or "") in {
            "semantic_edit",
            "regen_edit",
        }
        job = build_job(
            project_id=manifest.id,
            prompt=generation,
            user_prompt=user_prompt,
            generation_prompt=generation,
            semantic_regen=semantic,
            solidify_mm=solidify_mm,
        )
        return run_generation_job(orch, manifest, job)
    raise ValueError("Нет предыдущего промпта генерации — сначала создайте модель через чат")
