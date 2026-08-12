from __future__ import annotations

import logging
import shutil
import tempfile
import traceback
from pathlib import Path

from mesh_forge.config import AppConfig, LLMConfig, load_config, update_llm_settings
from mesh_forge.domain import GenerationJob, JobOptions
from mesh_forge.manifest import ProjectManifest, list_projects
from mesh_forge.mesh_qc import analyze_mesh, is_print_ready
from mesh_forge.orchestrator import Orchestrator
from mesh_forge import progress as prog
from mesh_forge.adapters import LMStudioClient

from api.schemas import (
    ArtifactInfo,
    ExportInfo,
    LLMModelsResponse,
    LLMSettings,
    OperationResult,
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


def operation_result(manifest: ProjectManifest, message: str) -> OperationResult:
    return OperationResult(
        message=message,
        project=project_detail(manifest),
        qc_report=qc_report_for(manifest),
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
) -> GenerationJob:
    return GenerationJob(
        project_id=project_id,
        prompt=prompt.strip(),
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
        ),
    )
