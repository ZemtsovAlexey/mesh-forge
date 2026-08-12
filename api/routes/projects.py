from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

from api.deps import get_orchestrator, load_project
from api.schemas import (
    ChatStateInfo,
    CreateProjectRequest,
    ExportInfo,
    OperationResult,
    ProgressInfo,
    ProjectDetail,
    ProjectSummary,
)
from api.services import (
    build_job,
    confirm_project_chat,
    export_info,
    get_project_chat,
    list_project_summaries,
    post_project_chat_message,
    project_detail,
    run_generation_job,
    save_upload_to_tmp,
)
from mesh_forge.manifest import create_project
from mesh_forge import progress as prog

router = APIRouter(prefix="/api/projects", tags=["projects"])


@router.get("", response_model=list[ProjectSummary])
def get_projects() -> list[ProjectSummary]:
    return list_project_summaries()


@router.post("", response_model=ProjectDetail, status_code=201)
def post_project(body: CreateProjectRequest) -> ProjectDetail:
    manifest = create_project(body.name.strip())
    return project_detail(manifest)


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

    try:
        data = await run_in_threadpool(
            post_project_chat_message,
            manifest,
            text=text,
            has_images=image_count > 0,
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
        # Legacy text+mesh: semantic regen via ComfyUI (not filter ops).
        job = build_job(
            project_id=manifest.id,
            prompt=prompt,
            user_prompt=prompt,
            generation_prompt=prompt,
            semantic_regen=True,
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
