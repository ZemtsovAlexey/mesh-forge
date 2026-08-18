from __future__ import annotations

import logging
from pathlib import Path

from pydantic_ai import RunContext

from mesh_forge.agent.deps import ChatDeps
from mesh_forge.chat.models import Artifact
from mesh_forge.ops.geometry import load_mesh, save_mesh
from mesh_forge.ops.topo import (
    parse_kind,
    topo_valid,
    topology_from_ids,
)

logger = logging.getLogger("mesh_forge.tools")


def resolve_topo(
    ctx: RunContext[ChatDeps],
    mesh,
    *,
    elem: str | None = None,
    vertex: int | None = None,
    face: int | None = None,
    edge: str | None = None,
    hops: int | None = None,
) -> dict | None:
    """Explicit ids win; else the stored click topology."""
    has_ids = (
        (vertex is not None and int(vertex) >= 0)
        or (face is not None and int(face) >= 0)
        or bool((edge or "").strip())
    )
    if has_ids:
        kind = elem or ("face" if face is not None else "vertex" if vertex is not None else "edge")
        topo = topology_from_ids(mesh, kind=kind, vertex=vertex, face=face, edge=edge)
        topo["hops"] = max(1, min(int(hops or 12), 24))
        return topo
    stored = ctx.deps.store.active_mesh_topo(ctx.deps.chat_id)
    if not topo_valid(stored):
        return None
    stored = dict(stored)
    if elem:
        stored["kind"] = parse_kind(elem)
    stored["hops"] = max(1, min(int(hops if hops is not None else stored.get("hops") or 12), 24))
    return stored


LOOK_AFTER = (
    "look(target='mesh') now, overview: no region, zoom 1. "
    "Compare to the user request. If it is worse — restore_mesh(to='previous')."
)


def carve_painted_mask(ctx: RunContext[ChatDeps], mesh, mesh_name: str):
    from mesh_forge.ops.edit import EditError
    from mesh_forge.ops.geometry import carve_faces

    drop = ctx.deps.store.load_mesh_mask(
        ctx.deps.chat_id, n_faces=int(len(mesh.faces)), mesh_name=mesh_name
    )
    if drop is None or not drop.any():
        raise EditError("No painted mask on this mesh. Call mask_mesh first.")
    if int(drop.sum()) < 12:
        raise EditError(
            "Painted mask is too small to be the extra bit (red would be invisible). "
            "Click the artifact on a look PNG and call mask_mesh again."
        )
    n = int(len(mesh.faces))
    if n > 0 and int(drop.sum()) > min(8000, max(48, int(0.08 * n))):
        raise EditError(
            "Painted mask is the skirt/body, not the extra bit. "
            "restore_mesh if you already deleted, then click the petal on a look PNG and mask_mesh again."
        )
    return carve_faces(mesh, drop, min_keep_ratio=0.50, min_keep_faces=8, drop_crumbs=False)


def resolve_edit_target(
    ctx: RunContext[ChatDeps],
    region: str | None,
) -> tuple[str, tuple[float, float, float, float, float, float], bool]:
    """Click target if present; otherwise the named region."""
    from mesh_forge.ops.region import RegionError, infer_region, parse_region, pick_box, region_box

    label, pick = ctx.deps.store.active_mesh_target(ctx.deps.chat_id)
    if len(pick) >= 3:
        nx, ny, nz = float(pick[0]), float(pick[1]), float(pick[2])
        radius = float(pick[3]) if len(pick) > 3 else 0.022
        hint = label or infer_region(nx, ny, nz)
        return f"pick:{hint}", pick_box(nx, ny, nz, radius), False
    raw = (region or "").strip()
    if raw:
        name = parse_region(raw)
        return name, region_box(name), True
    if label:
        name = parse_region(label)
        return name, region_box(name), True
    raise RegionError("Need region or a click on the mesh.")


def reply_image_id(ctx: RunContext[ChatDeps], *, prefer_front: bool = True) -> str | None:
    images = [a for a in ctx.deps.reply_artifacts if a.kind == "image"]
    if not images:
        return None
    if prefer_front:
        for art in images:
            if (art.view or art.label or "").strip().lower() == "front":
                return art.id
    return images[0].id


def resolve_mesh(ctx: RunContext[ChatDeps], mesh_ref: str | None = None) -> Path:
    store = ctx.deps.store
    chat_id = ctx.deps.chat_id
    if mesh_ref and mesh_ref.strip():
        return store.resolve_ref(chat_id, mesh_ref.strip())
    topo = store.active_mesh_topo(chat_id)
    mesh_name = str((topo or {}).get("mesh") or "").strip()
    if mesh_name:
        try:
            return store.resolve_ref(chat_id, mesh_name)
        except FileNotFoundError:
            pass
    current = store.current_mesh(chat_id)
    if current is None:
        raise FileNotFoundError("Нет текущего mesh. Сгенерируйте или прикрепите STL.")
    return current


def save_mesh_artifact(
    ctx: RunContext[ChatDeps],
    mesh,
    filename: str,
    *,
    label: str = "",
    tool_id: str = "",
    role: str = "edit",
    make_current: bool = True,
) -> Artifact:
    dest = ctx.deps.store.new_file(ctx.deps.chat_id, filename)
    save_mesh(mesh, dest)
    if make_current:
        ctx.deps.store.set_current_mesh(ctx.deps.chat_id, dest, role=role)
    art = ctx.deps.store.artifact_from_path(ctx.deps.chat_id, dest, label=label or dest.stem)
    ctx.deps.emit_artifact(art, tool_id=tool_id)
    emit_mesh_preview(ctx, dest, mesh=mesh)
    return art


def apply_saved_proposal_mesh(
    ctx: RunContext[ChatDeps],
    state: dict[str, str],
    *,
    filename: str = "removed.stl",
    label: str = "removed",
    role: str = "edit",
) -> Artifact:
    ref = str((state or {}).get("proposal_mesh") or "").strip()
    if not ref:
        raise FileNotFoundError("No saved proposal mesh.")
    path = ctx.deps.store.resolve_ref(ctx.deps.chat_id, ref)
    mesh = load_mesh(path)
    return save_mesh_artifact(
        ctx,
        mesh,
        filename,
        label=label,
        role=role,
        make_current=True,
    )


def emit_mesh_preview(ctx: RunContext[ChatDeps], mesh_path: Path, *, mesh=None) -> None:
    try:
        ctx.deps.store.ensure_mesh_preview(ctx.deps.chat_id, mesh_path, mesh=mesh)
    except Exception:
        logger.warning("mesh preview failed for %s", mesh_path.name, exc_info=True)


def emit_masked_mesh_view(
    ctx: RunContext[ChatDeps],
    src: Path,
    mesh,
    face_mask,
    *,
    tool_id: str = "",
    camera: str = "viewer",
    zoom: float = 1.5,
) -> Artifact:
    from mesh_forge.render import export_mask_preview_glb, render_mesh_preview

    glb = ctx.deps.store.new_file(ctx.deps.chat_id, "masked.glb")
    export_mask_preview_glb(mesh, face_mask, glb)
    preview = ctx.deps.store.mesh_preview_path(glb)
    render_mesh_preview(
        src,
        preview,
        size=512,
        mesh=mesh,
        face_mask=face_mask,
        camera=camera,
        zoom=zoom,
    )
    art = ctx.deps.store.artifact_from_path(ctx.deps.chat_id, glb, label="маска · 3D")
    ctx.deps.emit_artifact(art, tool_id=tool_id)
    png = ctx.deps.store.new_file(ctx.deps.chat_id, "masked_view.png")
    render_mesh_preview(
        src,
        png,
        size=768,
        mesh=mesh,
        face_mask=face_mask,
        camera=camera,
        zoom=zoom,
    )
    png_art = ctx.deps.store.artifact_from_path(
        ctx.deps.chat_id, png, label="маска · вид", view=camera
    )
    ctx.deps.emit_artifact(png_art, tool_id=tool_id)
    return art


def save_image_artifact(
    ctx: RunContext[ChatDeps],
    src: Path,
    *,
    label: str = "",
    view: str = "",
    tool_id: str = "",
) -> Artifact:
    dest = ctx.deps.store.new_file(ctx.deps.chat_id, f"{view or src.stem}{src.suffix or '.png'}")
    dest.write_bytes(src.read_bytes())
    art = ctx.deps.store.artifact_from_path(
        ctx.deps.chat_id, dest, label=label or view or dest.stem, view=view
    )
    ctx.deps.emit_artifact(art, tool_id=tool_id)
    return art


def load_current_or_ref(ctx: RunContext[ChatDeps], mesh_ref: str | None = None):
    return load_mesh(resolve_mesh(ctx, mesh_ref))
