from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

from api.deps import get_orchestrator, load_project
from api.schemas import (
    ChatStateInfo,
    CreateProjectRequest,
    DuplicateProjectRequest,
    ExportInfo,
    OperationResult,
    PipelineRedoRequest,
    PipelineStateInfo,
    ProgressInfo,
    ProjectDetail,
    ProjectSummary,
    RenameProjectRequest,
)
from api.services import (
    build_job,
    confirm_project_chat,
    continue_project_pipeline,
    export_info,
    get_pipeline,
    get_project_chat,
    list_project_summaries,
    post_project_chat_message,
    project_detail,
    redo_project_pipeline,
    regenerate_project,
    restart_project_chat_from_message,
    run_generation_job,
    save_upload_to_tmp,
)
from mesh_forge.manifest import create_project, delete_project, duplicate_project, rename_project
from mesh_forge import progress as prog

router = APIRouter(prefix="/api/projects", tags=["projects"])


@router.get("", response_model=list[ProjectSummary])
def get_projects() -> list[ProjectSummary]:
    return list_project_summaries()


@router.post("", response_model=ProjectDetail, status_code=201)
def post_project(body: CreateProjectRequest) -> ProjectDetail:
    manifest = create_project(body.name.strip())
    return project_detail(manifest)


@router.patch("/{project_id}", response_model=ProjectDetail)
def patch_project(project_id: str, body: RenameProjectRequest) -> ProjectDetail:
    try:
        manifest = load_project(project_id)
        return project_detail(rename_project(manifest, body.name))
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/{project_id}/duplicate", response_model=ProjectDetail, status_code=201)
def post_duplicate_project(project_id: str, body: DuplicateProjectRequest | None = None) -> ProjectDetail:
    try:
        manifest = load_project(project_id)
        name = body.name if body else None
        return project_detail(duplicate_project(manifest, name=name))
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc


@router.delete("/{project_id}", status_code=204)
def remove_project(project_id: str) -> None:
    try:
        delete_project(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.get("/{project_id}", response_model=ProjectDetail)
def get_project(project_id: str) -> ProjectDetail:
    try:
        return project_detail(load_project(project_id))
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.get("/{project_id}/mesh")
def get_project_mesh(project_id: str):
    try:
        manifest = load_project(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    mesh = manifest.current_mesh_path()
    if not mesh or not mesh.is_file():
        raise HTTPException(404, "No mesh for this project")
    media = "model/stl" if mesh.suffix.lower() == ".stl" else "application/octet-stream"
    return FileResponse(mesh, media_type=media, filename=mesh.name)


@router.get("/{project_id}/artifacts/{version}/{filename}")
def get_project_artifact(project_id: str, version: int, filename: str):
    try:
        manifest = load_project(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    safe_name = Path(filename).name
    if not safe_name or safe_name != filename or ".." in filename:
        raise HTTPException(400, "Invalid artifact name")
    path = (manifest.root / "models" / f"v{version}" / "artifacts" / safe_name).resolve()
    root = (manifest.root / "models").resolve()
    if not str(path).startswith(str(root)) or not path.is_file():
        raise HTTPException(404, "Artifact not found")
    suffix = path.suffix.lower()
    media = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".stl": "model/stl",
        ".glb": "model/gltf-binary",
    }.get(suffix, "application/octet-stream")
    return FileResponse(path, media_type=media, filename=path.name)


@router.get("/{project_id}/media/{file_path:path}")
def get_project_media(project_id: str, file_path: str):
    try:
        manifest = load_project(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    from mesh_forge.application.chat_results import resolve_media_path

    try:
        path = resolve_media_path(manifest, file_path)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    suffix = path.suffix.lower()
    media = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".stl": "model/stl",
        ".obj": "text/plain",
        ".glb": "model/gltf-binary",
    }.get(suffix, "application/octet-stream")
    return FileResponse(path, media_type=media, filename=path.name)


@router.get("/{project_id}/export", response_model=ExportInfo)
def get_project_export(project_id: str) -> ExportInfo:
    try:
        return export_info(load_project(project_id))
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.get("/{project_id}/progress", response_model=ProgressInfo)
def get_project_progress(project_id: str) -> ProgressInfo:
    state = prog.get(project_id)
    if not state:
        return ProgressInfo(
            project_id=project_id,
            operation="",
            percent=0,
            stage="idle",
            active=False,
            elapsed_sec=0,
        )
    return ProgressInfo(
            project_id=state.project_id,
            operation=state.operation,
            percent=state.percent,
            stage=state.stage,
            active=state.active,
            error=state.error,
            elapsed_sec=state.to_dict()["elapsed_sec"],
        )


@router.post("/{project_id}/cancel")
def post_project_cancel(project_id: str) -> dict:
    """Request stop for the active chat/Comfy job."""
    try:
        load_project(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    active = prog.request_cancel(project_id)
    try:
        from mesh_forge.adapters import ComfyUiClient

        ComfyUiClient().interrupt()
    except Exception:
        pass
    return {"ok": True, "was_active": active, "message": "Остановка запрошена"}


@router.get("/{project_id}/chat", response_model=ChatStateInfo)
def get_chat(project_id: str) -> ChatStateInfo:
    try:
        data = get_project_chat(load_project(project_id))
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    return ChatStateInfo(**data)


@router.post("/{project_id}/chat/messages", response_model=ChatStateInfo)
async def post_chat_message(
    project_id: str,
    text: str = Form(""),
    ref_ids: str = Form(""),
    images: list[UploadFile] | None = File(None),
) -> ChatStateInfo:
    try:
        manifest = load_project(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc

    upload_dir = manifest.root / "work" / "chat_uploads"
    await run_in_threadpool(lambda: upload_dir.mkdir(parents=True, exist_ok=True))
    # Replace previous pending uploads for this chat turn.
    if upload_dir.is_dir():
        for old in upload_dir.glob("*"):
            if old.is_file():
                old.unlink(missing_ok=True)

    image_count = 0
    for index, upload in enumerate(images or [], start=1):
        if upload and upload.filename:
            suffix = Path(upload.filename).suffix or ".png"
            dest = upload_dir / f"image_{index}{suffix}"
            content = await upload.read()
            dest.write_bytes(content)
            image_count += 1

    refs = [part.strip() for part in (ref_ids or "").replace(";", ",").split(",") if part.strip()]

    try:
        data = await run_in_threadpool(
            post_project_chat_message,
            manifest,
            text=text,
            has_images=image_count > 0,
            ref_ids=refs,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc
    return ChatStateInfo(**data)


@router.post("/{project_id}/chat/restart", response_model=ChatStateInfo)
async def post_chat_restart(
    project_id: str,
    message_id: str = Form(...),
) -> ChatStateInfo:
    try:
        manifest = load_project(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    try:
        data = await run_in_threadpool(
            restart_project_chat_from_message,
            manifest,
            message_id,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc
    return ChatStateInfo(**data)


@router.post("/{project_id}/chat/confirm", response_model=OperationResult)
async def post_chat_confirm(
    project_id: str,
    images: list[UploadFile] | None = File(None),
    solidify_mm: float = Form(0.0),
    mode: str = Form("light"),
    smooth_iters: int = Form(1),
    remove_bg: bool = Form(True),
) -> OperationResult:
    try:
        manifest = load_project(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc

    image_paths: list[Path] = []
    for upload in images or []:
        if upload and upload.filename:
            suffix = Path(upload.filename).suffix or ".png"
            staged = await run_in_threadpool(save_upload_to_tmp, upload, suffix)
            image_paths.append(staged)

    if not image_paths:
        upload_dir = manifest.root / "work" / "chat_uploads"
        if upload_dir.is_dir():
            image_paths = sorted(
                [p for p in upload_dir.iterdir() if p.is_file()],
                key=lambda p: p.name,
            )

    try:
        def _confirm():
            return confirm_project_chat(
                get_orchestrator(),
                manifest,
                image_paths=image_paths,
                solidify_mm=solidify_mm,
                mode=mode,
                smooth_iters=smooth_iters,
                remove_bg=remove_bg,
            )

        return await run_in_threadpool(_confirm)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(500, str(exc)) from exc


@router.get("/{project_id}/pipeline", response_model=PipelineStateInfo)
def get_project_pipeline(project_id: str) -> PipelineStateInfo:
    try:
        manifest = load_project(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    return PipelineStateInfo(**get_pipeline(manifest))


@router.post("/{project_id}/pipeline/continue", response_model=OperationResult)
async def post_pipeline_continue(project_id: str) -> OperationResult:
    try:
        manifest = load_project(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    try:
        return await run_in_threadpool(lambda: continue_project_pipeline(manifest))
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(500, str(exc)) from exc


@router.post("/{project_id}/pipeline/redo", response_model=OperationResult)
async def post_pipeline_redo(project_id: str, body: PipelineRedoRequest | None = None) -> OperationResult:
    try:
        manifest = load_project(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    payload = body or PipelineRedoRequest()
    try:
        return await run_in_threadpool(
            lambda: redo_project_pipeline(
                manifest,
                step=payload.step,
                brief_en=payload.brief_en,
            )
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(500, str(exc)) from exc


@router.get("/{project_id}/pipeline/image/{stage}/{label}")
def get_pipeline_image(project_id: str, stage: str, label: str) -> FileResponse:
    try:
        manifest = load_project(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    from mesh_forge.application.stepped_pipeline import resolve_pipeline_image

    try:
        path = resolve_pipeline_image(manifest, stage, label)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    return FileResponse(path)


@router.post("/{project_id}/regenerate", response_model=OperationResult)
async def post_regenerate(
    project_id: str,
    solidify_mm: float = Form(0.0),
) -> OperationResult:
    try:
        manifest = load_project(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc

    try:
        return await run_in_threadpool(
            lambda: regenerate_project(
                get_orchestrator(),
                manifest,
                solidify_mm=solidify_mm,
            )
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(500, str(exc)) from exc


@router.post("/{project_id}/jobs", response_model=OperationResult)
async def post_job(
    project_id: str,
    prompt: str = Form(""),
    images: list[UploadFile] | None = File(None),
    mesh: UploadFile | None = File(None),
    use_current_mesh: bool = Form(False),
    backend: str = Form("auto"),
    remove_bg: bool = Form(True),
    solidify_mm: float = Form(0.0),
    mode: str = Form("light"),
    smooth_iters: int = Form(1),
) -> OperationResult:
    try:
        manifest = load_project(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    image_paths: list[Path] = []
    for upload in images or []:
        if upload and upload.filename:
            suffix = Path(upload.filename).suffix or ".png"
            staged = await run_in_threadpool(save_upload_to_tmp, upload, suffix)
            image_paths.append(staged)

    source_mesh = None
    if mesh and mesh.filename:
        suffix = Path(mesh.filename).suffix or ".stl"
        source_mesh = await run_in_threadpool(save_upload_to_tmp, mesh, suffix)
    elif use_current_mesh:
        source_mesh = manifest.current_mesh_path()

    # Advanced manual cleanup: mesh only, no prompt/images.
    if source_mesh and not prompt.strip() and not image_paths:
        job = build_job(
            project_id=manifest.id,
            source_mesh=source_mesh,
            use_current_mesh=use_current_mesh,
            solidify_mm=solidify_mm,
            scan_mode=mode,
            smooth_iters=smooth_iters,
        )
    elif source_mesh and prompt.strip() and not image_paths:
        # Text+mesh: geometry edit on current mesh (semantic regen is chat-confirm only).
        job = build_job(
            project_id=manifest.id,
            prompt=prompt,
            user_prompt=prompt,
            generation_prompt=prompt,
            source_mesh=source_mesh,
            use_current_mesh=True,
            semantic_regen=False,
            solidify_mm=solidify_mm,
        )
    else:
        if not prompt.strip() and not image_paths and not source_mesh:
            raise HTTPException(400, "Provide text, images, or a mesh")
        job = build_job(
            project_id=manifest.id,
            prompt=prompt,
            user_prompt=prompt,
            generation_prompt=prompt,
            image_paths=image_paths,
            source_mesh=source_mesh,
            use_current_mesh=use_current_mesh,
            backend=backend,
            remove_bg=remove_bg,
            solidify_mm=solidify_mm,
            scan_mode=mode,
            smooth_iters=smooth_iters,
        )

    try:
        def _do_job():
            return run_generation_job(get_orchestrator(), manifest, job)

        return await run_in_threadpool(_do_job)
    except RuntimeError as exc:
        raise HTTPException(500, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
