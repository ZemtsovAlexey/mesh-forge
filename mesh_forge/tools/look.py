from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

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
_NAMED_YAW = {
    "front": 0.0,
    "right": 90.0,
    "back": 180.0,
    "left": -90.0,
    "viewer": 45.0,
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


class LookShot(NamedTuple):
    camera: str
    zoom: float
    region: str
    yaw: float | None = None
    pitch: float | None = None
    shift: tuple[float, float, float] = (0.0, 0.0, 0.0)


def _clamp_zoom(zoom: float) -> float:
    return max(0.55, min(float(zoom or 1.0), 5.0))


def _clamp_shift(value: float) -> float:
    return max(-1.0, min(1.0, float(value)))


def _is_yaw_token(part: str) -> bool:
    if not part or part in {"3/4"}:
        return False
    try:
        float(part)
    except ValueError:
        return False
    return True


_MASK_PACK = {
    "orbit",
    "around",
    "all",
    "вокруг",
    "орбита",
    "detail",
    "closeup",
    "крупно",
    "деталь",
}
_MASK_OPPOSITE = {
    "right": "left",
    "left": "right",
    "front": "back",
    "back": "front",
    "top": "front",
    "viewer": "right",
}


def default_mesh_look(
    views: str = "",
    region: str = "",
    question: str = "",
    *,
    pick: bool = False,
    free_camera: bool = False,
    mask: bool = False,
    start_view: str = "",
) -> tuple[str, str]:
    """Compare to the user request after edits unless the caller already picked shots."""
    if mask:
        raw = (views or "").strip()
        first = raw.split(",")[0].strip().lower() if raw else ""
        packed = first in _MASK_PACK
        if free_camera and not raw:
            pass
        elif not first or packed:
            views = (start_view or "right").split(",")[0].strip() or "right"
        else:
            views = raw.split(",")[0].strip()
    elif not (views or "").strip() and not (region or "").strip() and not free_camera:
        views = "front,left,right" if pick else "viewer,left,right"
    if not (question or "").strip():
        if pick:
            question = "Оранжевая точка — клик. В workspace есть face/vertex/edge этого клика."
        else:
            question = ""
    return views, question


def _mask_next_camera(current: str, seen: list[str]) -> str:
    cam = (current or "right").split(",")[0].strip().lower() or "right"
    looked = {str(v).strip().lower() for v in seen if v}
    opp = _MASK_OPPOSITE.get(cam, "front")
    if opp not in looked:
        return opp
    for name in ("right", "left", "front", "back", "top", "viewer"):
        if name not in looked and name != cam:
            return name
    return opp


def _mask_review_question(question: str, *, camera: str, seen: list[str]) -> str:
    cam = (camera or "right").split(",")[0].strip() or "right"
    prior = [str(v) for v in seen if v and str(v) != cam]
    bits = [f"Сейчас кадр с маской: {cam}."]
    if prior:
        bits.append(f"Уже смотрели: {', '.join(prior)}. Если сравнение с предыдущими видами полезно, упомяни это словами.")
    else:
        bits.append("Других ракурсов ещё не было. Оцени только видимый overlay, без команд NEXT.")
    extra = " ".join(bits)
    return f"{question}\n{extra}".strip() if question else extra


def parse_look_shots(
    views: str = "",
    *,
    zoom: float = 1.0,
    region: str = "",
    yaw: float | None = None,
    pitch: float | None = None,
    shift: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> list[LookShot]:
    """Named cameras and/or yaw degrees."""
    zoom = _clamp_zoom(zoom)
    shift = (
        _clamp_shift(shift[0] if len(shift) > 0 else 0.0),
        _clamp_shift(shift[1] if len(shift) > 1 else 0.0),
        _clamp_shift(shift[2] if len(shift) > 2 else 0.0),
    )
    region = _REGION_ALIASES.get((region or "").strip().lower(), (region or "").strip().lower())
    if region and region not in _REGION_LABELS:
        region = ""
    raw = (views or "").strip().lower()
    if region and not raw:
        close_zoom = zoom if zoom > 1.05 else 2.2
        cam = "custom" if yaw is not None or pitch is not None else "viewer"
        return [
            LookShot(cam, 1.0, "", yaw, pitch, shift),
            LookShot(cam, close_zoom, region, yaw, pitch, shift),
        ]
    if region and region not in {"", "center"} and zoom <= 1.05:
        zoom = 2.2
    tokens: list[str] = []
    if not raw:
        tokens = ["custom"] if yaw is not None or pitch is not None else ["viewer"]
    else:
        parts = [p.strip() for p in raw.replace(";", ",").replace("|", ",").split(",") if p.strip()]
        for part in parts:
            if part in {"orbit", "around", "all", "вокруг", "орбита"}:
                if yaw is not None:
                    tokens.extend([f"{float(yaw) + d:g}" for d in (0.0, 90.0, 180.0, -90.0)])
                else:
                    tokens.extend(_ORBIT)
                continue
            if part in {"detail", "closeup", "крупно", "деталь"}:
                if region:
                    tokens.append("viewer")
                else:
                    tokens.extend(["viewer", "front", "top"])
                continue
            tokens.append(_CAMERA_ALIASES.get(part, part))
    seen: set[str] = set()
    out: list[LookShot] = []
    for token in tokens:
        if _is_yaw_token(token):
            shot_yaw = float(token)
            key = f"yaw:{shot_yaw:.1f}:{zoom:.2f}:{region}:{shift}"
            if key in seen:
                continue
            seen.add(key)
            out.append(LookShot("custom", zoom, region, shot_yaw, pitch, shift))
        else:
            if token not in _CAMERAS and token != "custom":
                continue
            if token == "custom":
                out.append(LookShot("custom", zoom, region, yaw, pitch, shift))
            elif yaw is not None and token in _NAMED_YAW:
                shot_yaw = float(yaw) + _NAMED_YAW[token]
                out.append(LookShot("custom", zoom, region, shot_yaw, pitch, shift))
            elif yaw is not None and token == "top":
                out.append(LookShot("top", zoom, region, float(yaw), 78.0 if pitch is None else pitch, shift))
            else:
                out.append(LookShot(token, zoom, region, None, pitch, shift))
        if len(out) >= _MAX_FRAMES:
            break
    out = _unique_shots(out)
    return out or [LookShot("viewer", zoom, region, yaw, pitch, shift)]


def _unique_shots(shots: list[LookShot]) -> list[LookShot]:
    """Drop frames that would render as the same camera pose."""
    seen: set[tuple] = set()
    uniq: list[LookShot] = []
    for shot in shots:
        if shot.yaw is not None or shot.camera == "custom":
            y = 0.0 if shot.yaw is None else round(float(shot.yaw), 1)
            p = 15.0 if shot.pitch is None else round(float(shot.pitch), 1)
            pose: tuple = ("yp", y, p)
        else:
            pose = ("cam", shot.camera, None if shot.pitch is None else round(float(shot.pitch), 1))
        key = (
            pose,
            round(shot.zoom, 2),
            shot.region,
            tuple(round(v, 3) for v in shot.shift),
        )
        if key in seen:
            continue
        seen.add(key)
        uniq.append(shot)
    return uniq


def _shot_caption(shot: LookShot, *, pick: bool = False, mask: bool = False) -> str:
    if shot.camera == "custom" or shot.yaw is not None:
        yaw = 0.0 if shot.yaw is None else shot.yaw
        pitch = 15.0 if shot.pitch is None else shot.pitch
        bits = [f"yaw {yaw:.0f}°", f"pitch {pitch:.0f}°"]
    else:
        bits = [_CAMERA_LABELS.get(shot.camera, shot.camera)]
        if shot.pitch is not None:
            bits.append(f"pitch {shot.pitch:.0f}°")
    if mask:
        bits.append("маска · красное удалится")
    elif pick:
        bits.append("клик")
    elif shot.region and shot.region not in {"", "center"}:
        bits.append(_REGION_LABELS.get(shot.region, shot.region))
    if any(abs(v) > 0.02 for v in shot.shift):
        bits.append("сдвиг")
    if shot.zoom >= 1.4:
        bits.append("крупно")
    elif shot.zoom <= 0.85:
        bits.append("далеко")
    return " · ".join(bits)


def _vision_label(shot: LookShot, *, pick: bool = False) -> str:
    if shot.camera == "custom" or shot.yaw is not None:
        yaw = 0.0 if shot.yaw is None else shot.yaw
        pitch = 15.0 if shot.pitch is None else shot.pitch
        name = f"yaw {yaw:.0f} pitch {pitch:.0f}"
    else:
        name = shot.camera
        if shot.pitch is not None:
            name += f" pitch {shot.pitch:.0f}"
    if pick:
        name += " orange marker"
    elif shot.region and shot.region not in {"", "center"}:
        name += f" {shot.region}"
    if shot.zoom >= 1.4:
        name += " closeup"
    return name


def _prefer_named_images(
    images: list[tuple[str, Path]],
    views: str,
) -> list[tuple[str, Path]]:
    tokens = {
        _CAMERA_ALIASES.get(part.strip().lower(), part.strip().lower())
        for part in views.replace(";", ",").replace("|", ",").split(",")
        if part.strip()
    }
    if not tokens:
        return images
    named = [
        item
        for item in images
        if any(token in (item[0] or "").lower() for token in tokens)
    ]
    return named or images


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
        "custom": "камера",
        "зрение": "зрение",
        "рассуждение": "рассуждение",
        "описание": "описание",
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
        yaw: float | None = None,
        pitch: float | None = None,
        shift_x: float = 0.0,
        shift_y: float = 0.0,
        shift_z: float = 0.0,
    ) -> str:
        """Look at images or the mesh. Does not edit.

        target: auto|mesh|images. images = attached photos, no STL required.
        views: mesh cameras front|left|right|back|top|viewer|orbit, or yaw degrees like 20,90,-40.
        On target=images, views is a photo label (front), not a mesh camera.
        Several mesh angles can go in one views= list — except while a red mask is on:
        then one camera per look. Read NEXT: look <cam> and look that view next.
        Free camera: yaw+pitch with empty views = one shot. yaw plus views=left,right = those sides around that heading.
        pitch 0=level, +from above, -from below. zoom 0.7 farther, 1=whole, 2–4 closer.
        shift_x/y/z: move look-at, -1..1. If the user clicked, omit region.
        Photos end with NEXT: regen | cutout | mesh.
        """
        store = ctx.deps.store
        chat_id = ctx.deps.chat_id
        from mesh_forge import progress as prog

        prog.raise_if_cancelled(chat_id)
        wanted = (target or "auto").strip().lower()
        mesh_shots = bool(
            views.strip()
            or region.strip()
            or float(zoom or 1.0) > 1.05
            or yaw is not None
            or pitch is not None
            or abs(float(shift_x or 0)) > 0.01
            or abs(float(shift_y or 0)) > 0.01
            or abs(float(shift_z or 0)) > 0.01
        )
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
        elif wanted in {"auto", "images"}:
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
            if wanted == "images" and views.strip():
                image_refs = _prefer_named_images(image_refs, views)
        images: list[tuple[str, Path]] = list(image_refs)
        looking_at_mesh = False
        checking_mask = False
        want_mesh = wanted != "images" and (
            wanted == "mesh"
            or mesh_from_refs is not None
            or mesh_shots
            or (wanted == "auto" and not images)
        )
        if want_mesh:
            looking_at_mesh = True
            try:
                mesh_path = mesh_from_refs or resolve_mesh(ctx)
            except FileNotFoundError:
                if not images:
                    raise
                looking_at_mesh = False
        if looking_at_mesh:
            _, pick = store.active_mesh_target(chat_id)
            has_pick = len(pick) >= 3
            crop = "" if has_pick else region
            if has_pick and float(zoom or 1.0) <= 1.05:
                zoom = 2.3
            goal = ""
            try:
                from mesh_forge.agent.workspace import _goal_from_messages, _safe_messages

                goal = _goal_from_messages(_safe_messages(store, chat_id))
            except Exception:
                goal = ""
            if goal and goal not in (question or ""):
                question = f"{goal}\n{question}".strip() if question else goal
            overlay = None
            try:
                seated = load_render_mesh(mesh_path)
                overlay = store.load_mesh_mask(
                    chat_id, n_faces=int(len(seated.faces)), mesh_name=mesh_path.name
                )
            except Exception:
                overlay = None
            if overlay is not None and not overlay.any():
                overlay = None
            checking_mask = overlay is not None
            start_view = ""
            if overlay is not None:
                from mesh_forge.ops.topo import mask_aim_side

                side = mask_aim_side(goal or question, None)
                start_view = side if side in {"left", "right"} else "right"
            views, question = default_mesh_look(
                views,
                crop,
                question,
                pick=has_pick,
                free_camera=yaw is not None,
                mask=overlay is not None,
                start_view=start_view,
            )
            images = _render_mesh_looks(
                ctx,
                mesh_path,
                views=views,
                zoom=zoom,
                region=crop,
                pick=pick if has_pick else None,
                yaw=yaw,
                pitch=pitch,
                shift=(shift_x, shift_y, shift_z),
                face_mask=overlay,
                mask_overlay=overlay is not None,
            )
            store.set_look_view(chat_id, views=str(views or ""), yaw=yaw, pitch=pitch, zoom=zoom)
            if checking_mask:
                looked = [
                    str(v)
                    for v in (dict(store.get_meta(chat_id).look_view or {}).get("seen") or [])
                    if v
                ]
                question = _mask_review_question(question, camera=str(views or ""), seen=looked)
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
        from mesh_forge import progress as prog

        prog.raise_if_cancelled(chat_id)
        prog.update(chat_id, 40, "зрение")
        seen = [0]

        def on_delta(kind: str, text: str) -> None:
            if not text:
                return
            if kind == "replace":
                ctx.deps.emit_event("tool_text", text=text)
                return
            seen[0] += len(text)
            ctx.deps.emit_event(
                "tool_thinking_delta" if kind == "thinking" else "tool_text_delta",
                delta=text,
            )
            prog.update(
                chat_id,
                min(92.0, 42.0 + seen[0] / 220.0),
                "рассуждение" if kind == "thinking" else "описание",
            )

        note = LMStudioClient().inspect_images(
            vision_inputs[:4],
            question=question,
            kind="mask" if checking_mask else ("mesh" if looking_at_mesh else "auto"),
            on_delta=on_delta,
        )
        if not note:
            note = "Модель зрения вернула пустое описание."
            ctx.deps.emit_event("tool_text_delta", delta=note)
        return note


def _render_mesh_looks(
    ctx: RunContext[ChatDeps],
    mesh_path: Path,
    *,
    views: str,
    zoom: float,
    region: str,
    pick: list[float] | tuple[float, ...] | None = None,
    yaw: float | None = None,
    pitch: float | None = None,
    shift: tuple[float, float, float] = (0.0, 0.0, 0.0),
    face_mask=None,
    mask_overlay: bool = False,
    emit: bool = True,
) -> list[tuple[str, Path]]:
    store = ctx.deps.store
    chat_id = ctx.deps.chat_id
    has_pick = pick is not None and len(pick) >= 3 and face_mask is None
    shots = parse_look_shots(
        views,
        zoom=zoom,
        region="" if has_pick else region,
        yaw=yaw,
        pitch=pitch,
        shift=shift,
    )
    if mask_overlay or face_mask is not None:
        shots = shots[:1]
    mesh = load_render_mesh(mesh_path)
    frames: list[tuple[str, Path]] = []
    for i, shot in enumerate(shots):
        caption = _shot_caption(shot, pick=has_pick, mask=bool(mask_overlay or face_mask is not None))
        preview = store.new_file(chat_id, f"look_mesh_{i}_{shot.camera}.png")
        render_mesh_preview(
            mesh_path,
            preview,
            camera=shot.camera,
            zoom=shot.zoom,
            region="" if has_pick else shot.region,
            mesh=mesh,
            pick=pick if has_pick else None,
            yaw=shot.yaw,
            pitch=shot.pitch,
            shift=shot.shift,
            face_mask=face_mask,
        )
        art = store.artifact_from_path(chat_id, preview, label=caption, view=shot.camera)
        if emit:
            ctx.deps.emit_artifact(art)
        frames.append((_vision_label(shot, pick=has_pick), preview))
    return frames
