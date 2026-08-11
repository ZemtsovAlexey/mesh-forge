from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

from api.deps import get_orchestrator, load_project
from api.schemas import (
    CreateProjectRequest,
    ExportInfo,
    OperationResult,
    ProgressInfo,
    ProjectDetail,
    ProjectSummary,
    TextCreateRequest,
    TextEditRequest,
)
from api.services import (
    create_photo,
    create_scan,
    create_text,
    edit_photo,
    edit_text,
    export_info,
    list_project_summaries,
    project_detail,
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


@router.post("/{project_id}/photo", response_model=OperationResult)
async def post_photo(
    project_id: str,
    image: UploadFile = File(...),
    remove_bg: bool = Form(True),
    solidify_mm: float = Form(0.0),
) -> OperationResult:
    try:
        manifest = load_project(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    suffix = Path(image.filename or "upload.png").suffix or ".png"
    path = await run_in_threadpool(save_upload_to_tmp, image, suffix)
    try:
        def _do_photo():
            return create_photo(
                get_orchestrator(),
                manifest,
                path,
                remove_bg=remove_bg,
                solidify_mm=solidify_mm,
            )

        return await run_in_threadpool(_do_photo)
    except RuntimeError as exc:
        raise HTTPException(500, str(exc)) from exc


@router.post("/{project_id}/scan", response_model=OperationResult)
async def post_scan(
    project_id: str,
    scan: UploadFile = File(...),
    mode: str = Form("light"),
    smooth_iters: int = Form(1),
    solidify_mm: float = Form(0.0),
) -> OperationResult:
    try:
        manifest = load_project(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    suffix = Path(scan.filename or "upload.stl").suffix or ".stl"
    path = await run_in_threadpool(save_upload_to_tmp, scan, suffix)
    try:
        def _do_scan():
            return create_scan(
                get_orchestrator(),
                manifest,
                path,
                mode=mode,
                smooth_iters=smooth_iters,
                solidify_mm=solidify_mm,
            )

        return await run_in_threadpool(_do_scan)
    except RuntimeError as exc:
        raise HTTPException(500, str(exc)) from exc


@router.post("/{project_id}/text", response_model=OperationResult)
def post_text(project_id: str, body: TextCreateRequest) -> OperationResult:
    try:
        manifest = load_project(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    try:
        return create_text(get_orchestrator(), manifest, body.prompt.strip(), body.mode)
    except RuntimeError as exc:
        raise HTTPException(500, str(exc)) from exc


@router.post("/{project_id}/edit/text", response_model=OperationResult)
def post_edit_text(project_id: str, body: TextEditRequest) -> OperationResult:
    try:
        manifest = load_project(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    try:
        return edit_text(
            get_orchestrator(),
            manifest,
            body.instruction.strip(),
            body.solidify_mm,
        )
    except RuntimeError as exc:
        raise HTTPException(500, str(exc)) from exc


@router.post("/{project_id}/edit/photo", response_model=OperationResult)
async def post_edit_photo(
    project_id: str,
    instruction: str = Form(""),
    reference: UploadFile | None = File(None),
) -> OperationResult:
    try:
        manifest = load_project(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    ref_path = None
    if reference and reference.filename:
        ref_path = await run_in_threadpool(
            save_upload_to_tmp, reference, Path(reference.filename).suffix or ".png"
        )
    try:
        def _do_edit():
            return edit_photo(
                get_orchestrator(),
                manifest,
                instruction.strip() or "match reference",
                ref_path,
            )

        return await run_in_threadpool(_do_edit)
    except RuntimeError as exc:
        raise HTTPException(500, str(exc)) from exc
