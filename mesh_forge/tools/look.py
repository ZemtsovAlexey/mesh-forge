from __future__ import annotations

from pathlib import Path

from pydantic_ai import RunContext

from mesh_forge.adapters import LMStudioClient
from mesh_forge.agent.deps import ChatDeps
from mesh_forge.render import load_render_mesh, render_mesh_preview
from mesh_forge.tools.base import MeshTool
from mesh_forge.tools.common import resolve_mesh

_MAX_FRAMES = 4
_CAMERAS = ("viewer", "front", "left", "right", "back", "top")
_ORBIT = ("front", "left", "back", "right")
_CAMERA_ALIASES = {
    "overview": "viewer",
    "3/4": "viewer",
    "viewer": "viewer",
    "обзор": "viewer",
    "front": "front",
    "спереди": "front",
    "left": "left",
    "слева": "left",
    "right": "right",
    "справа": "right",
    "back": "back",
    "сзади": "back",
    "top": "top",
    "сверху": "top",
}
_REGION_ALIASES = {
    "center": "center",
    "центр": "center",
    "top": "top",
    "верх": "top",
    "bottom": "bottom",
    "низ": "bottom",
    "legs": "legs",
    "ножки": "legs",
    "ноги": "legs",
    "seat": "seat",
    "сиденье": "seat",
    "left": "left",
    "слева": "left",
    "right": "right",
    "справа": "right",
    "front": "front",
    "спереди": "front",
    "back": "back",
    "сзади": "back",
    "backrest": "backrest",
    "спинка": "backrest",
    "резьба": "backrest",
}
_CAMERA_LABELS = {
    "viewer": "обзор 3/4",
    "front": "спереди",
    "left": "слева",
    "right": "справа",
    "back": "сзади",
    "top": "сверху",
}
_REGION_LABELS = {
    "center": "центр",
    "top": "верх",
    "bottom": "низ",
    "legs": "ножки",
    "seat": "сиденье",
    "left": "слева",
    "right": "справа",
    "front": "спереди",
    "back": "сзади",
    "backrest": "спинка",
}


def default_mesh_look(
    views: str = "",
    region: str = "",
    question: str = "",
) -> tuple[str, str]:
    """Compare left/right after edits unless the caller already picked shots."""
    if not (views or "").strip() and not (region or "").strip():
        views = "viewer,left,right"
    if not (question or "").strip():
        question = (
            "Сравни левый и правый бок. Если подлокотник, нога или сиденье срезаны "
            "или один бок короче — restore_mesh. Не называй срез нормой реконструкции."
        )
    return views, question


def parse_look_shots(
    views: str = "",
    *,
    zoom: float = 1.0,
    region: str = "",
) -> list[tuple[str, float, str]]:
    """Canonical (camera, zoom, region) shots, at most 4."""
    zoom = max(1.0, min(float(zoom or 1.0), 4.0))
    region = _REGION_ALIASES.get((region or "").strip().lower(), (region or "").strip().lower())
    if region and region not in _REGION_LABELS:
        region = ""
    if region and region not in {"", "center"} and zoom <= 1.05:
        zoom = 2.2
    raw = (views or "").strip().lower()
    cameras: list[str] = []
    if not raw:
        cameras = ["viewer"]
    else:
        parts = [p.strip() for p in raw.replace(";", ",").replace("|", ",").split(",") if p.strip()]
        for part in parts:
            if part in {"orbit", "around", "all", "вокруг", "орбита"}:
                cameras.extend(_ORBIT)
                continue
            if part in {"detail", "closeup", "крупно", "деталь"}:
                if region:
                    cameras.append("viewer")
                else:
                    cameras.extend(["viewer", "front", "top"])
                continue
            cameras.append(_CAMERA_ALIASES.get(part, part))
    seen: set[str] = set()
    out: list[tuple[str, float, str]] = []
    for camera in cameras:
        if camera not in _CAMERAS or camera in seen:
            continue
        seen.add(camera)
        out.append((camera, zoom, region))
        if len(out) >= _MAX_FRAMES:
            break
    return out or [("viewer", zoom, region)]


def _shot_caption(camera: str, zoom: float, region: str) -> str:
    label = _CAMERA_LABELS.get(camera, camera)
    bits = [label]
    if region and region not in {"", "center"}:
        bits.append(_REGION_LABELS.get(region, region))
    if zoom >= 1.4:
        bits.append("крупно")
    return " · ".join(bits)


def _vision_label(camera: str, zoom: float, region: str) -> str:
    name = f"mesh {camera}"
    if region and region not in {"", "center"}:
        name += f" {region}"
    if zoom >= 1.4:
        name += " closeup"
    return name


class Look(MeshTool):
    title = "Смотрю"
    heavy = True
    stages = {
        **MeshTool.stages,
        "viewer": "обзор 3/4",
        "front": "спереди",
        "left": "слева",
        "right": "справа",
        "back": "сзади",
        "top": "сверху",
        "orbit": "вокруг",
        "detail": "деталь",
    }

    def run(
        self,
        ctx: RunContext[ChatDeps],
        target: str = "auto",
        question: str = "",
        refs: list[str] | None = None,
        views: str = "",
        zoom: float = 1.0,
        region: str = "",
    ) -> str:
        """Vision look at images or a mesh. target: auto|mesh|images. refs: artifact ids.

        Mesh cameras — views: viewer (3/4) | front | left | right | back | top |
        orbit (4 sides) | comma list. Default mesh look is viewer+left+right to compare sides.
        zoom 1=whole object, 2–3=closer. region: top|bottom|legs|
        seat|backrest|left|right|front|back for a local detail. question: what to check.
        After carve/smooth/repair always compare left vs right. If a part was chopped, restore_mesh.
        For photos the last line is NEXT: regen | cutout | mesh. Follow that; do not images_to_mesh on regen/cutout.
        For a mesh: no NEXT. If a recent edit ruined the shape, restore_mesh; do not generate_image.
        """
        store = ctx.deps.store
        chat_id = ctx.deps.chat_id
        wanted = (target or "auto").strip().lower()
        mesh_shots = bool(views.strip() or region.strip() or float(zoom or 1.0) > 1.05)
        image_refs: list[tuple[str, Path]] = []
        mesh_from_refs: Path | None = None
        if refs:
            for ref in refs[:4]:
                try:
                    path = store.resolve_ref(chat_id, ref)
                except FileNotFoundError:
                    continue
                if path.suffix.lower() in {".stl", ".obj"}:
                    mesh_from_refs = mesh_from_refs or path
                else:
                    image_refs.append((ref, path))
        elif wanted in {"auto", "images"} and not mesh_shots:
            attached = [a for a in ctx.deps.attachments if a.kind == "image"]
            for art in attached[:4]:
                image_refs.append((art.label or art.id, store.resolve_file(chat_id, art.name)))
            if not image_refs:
                for art in [a for a in ctx.deps.reply_artifacts if a.kind in {"image", "mesh_preview"}][:4]:
                    try:
                        image_refs.append((art.label or art.view or art.id, store.resolve_ref(chat_id, art.id)))
                    except FileNotFoundError:
                        continue
            if not image_refs:
                pics = [a for a in store.list_files(chat_id) if a.kind == "image"][-4:]
                for art in pics:
                    image_refs.append((art.label or art.id, store.resolve_file(chat_id, art.name)))
        images: list[tuple[str, Path]] = list(image_refs)
        looked_at_mesh = False
        if wanted == "mesh" or mesh_shots or mesh_from_refs is not None or (wanted == "auto" and not images):
            mesh_path = mesh_from_refs or resolve_mesh(ctx)
            views, question = default_mesh_look(views, region, question)
            images = _render_mesh_looks(ctx, mesh_path, views=views, zoom=zoom, region=region)
            looked_at_mesh = True
        if not images:
            return "Нечего смотреть: нет картинок и нет mesh."
        vision_inputs: list[tuple[str, Path]] = []
        for label, path in images:
            if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
                vision_inputs.append((label, path))
            elif path.suffix.lower() in {".stl", ".obj"}:
                preview = store.new_file(chat_id, f"look_{path.stem}.png")
                render_mesh_preview(path, preview)
                art = store.artifact_from_path(chat_id, preview, label="обзор 3/4", view="preview")
                ctx.deps.emit_artifact(art)
                vision_inputs.append(("mesh viewer", preview))
        if not vision_inputs:
            return "Нечего смотреть."
        note = LMStudioClient().inspect_images(vision_inputs[:4], question=question)
        if not note:
            note = "Модель зрения вернула пустое описание."
        if looked_at_mesh:
            note += (
                "\nЕсли срезана деталь или один бок короче — restore_mesh(to='previous' или 'source'). "
                "Не хвали. Не generate_image."
            )
        return note


def _render_mesh_looks(
    ctx: RunContext[ChatDeps],
    mesh_path: Path,
    *,
    views: str,
    zoom: float,
    region: str,
) -> list[tuple[str, Path]]:
    store = ctx.deps.store
    chat_id = ctx.deps.chat_id
    shots = parse_look_shots(views, zoom=zoom, region=region)
    mesh = load_render_mesh(mesh_path)
    frames: list[tuple[str, Path]] = []
    for camera, shot_zoom, shot_region in shots:
        caption = _shot_caption(camera, shot_zoom, shot_region)
        preview = store.new_file(chat_id, f"look_mesh_{camera}.png")
        render_mesh_preview(
            mesh_path,
            preview,
            camera=camera,
            zoom=shot_zoom,
            region=shot_region,
            mesh=mesh,
        )
        art = store.artifact_from_path(chat_id, preview, label=caption, view=camera)
        ctx.deps.emit_artifact(art)
        frames.append((_vision_label(camera, shot_zoom, shot_region), preview))
    return frames
