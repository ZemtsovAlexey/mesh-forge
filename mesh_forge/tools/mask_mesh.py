from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic_ai import RunContext

from mesh_forge.agent.deps import ChatDeps
from mesh_forge.ops.geometry import load_mesh
from mesh_forge.ops.topo import (
    TopoError,
    dilate_face_mask,
    erode_face_mask,
    face_mask_for_topo,
    grow_visible_lump,
    mask_geometry_metrics,
    mask_aim_side,
    mask_from_view_observations,
    mask_is_tiny,
)
from mesh_forge.tools.base import MeshTool
from mesh_forge.tools.common import emit_masked_mesh_view, resolve_mesh
from mesh_forge.tools.select_mesh import inherit_look_camera

logger = logging.getLogger(__name__)

_View = Literal["viewer", "front", "left", "right", "back", "top"]
_Elem = Literal["vertex", "edge", "face"]
_Adjust = Literal["", "shrink", "grow", "retry"]


class MaskMesh(MeshTool):
    title = "Маска"
    heavy = True

    def run(
        self,
        ctx: RunContext[ChatDeps],
        views: _View | str | None = None,
        x: float | None = None,
        y: float | None = None,
        hops: int | None = None,
        elem: _Elem = "face",
        yaw: float | None = None,
        pitch: float | None = None,
        zoom: float = 1.0,
        apply: bool = False,
        mesh_ref: str | None = None,
        describe: str | None = None,
        adjust: _Adjust | str | None = None,
    ) -> str:
        """Paint the extra bit red: vision box on the look PNG, then adjacency fill inside that box.

        adjust: shrink / grow / retry after look said NEXT: mask shrink|grow|retry.
        """
        src = resolve_mesh(ctx, mesh_ref)
        loaded = load_mesh(src)
        _ = hops
        _ = elem
        _ = x
        _ = y
        hint = _goal_hint(ctx, describe)
        action = str(adjust or "").strip().lower()
        verdict: dict = {}
        try:
            focus_view, yaw, pitch, zoom = _resolve_focus_view(
                ctx, views=views, yaw=yaw, pitch=pitch, zoom=zoom, hint=hint
            )
            if action in {"shrink", "grow"}:
                mask = ctx.deps.store.load_mesh_mask(
                    ctx.deps.chat_id, n_faces=int(len(loaded.faces)), mesh_name=src.name
                )
                if mask is None or not mask.any():
                    return (
                        "mask_mesh adjust needs an automatic proposal first. "
                        "Call mask_mesh(describe=...) to rebuild the candidate mask."
                    )
                from mesh_forge.render import _seat_for_viewer

                verts, faces, _ = _seat_for_viewer(loaded)
                if action == "shrink":
                    mask = erode_face_mask(loaded, mask, hops=2)
                    note = "Auto-refine: shrunk the red overlay. "
                else:
                    mask = dilate_face_mask(
                        loaded, mask, hops=2, seated_verts=verts, seated_faces=faces
                    )
                    note = "Auto-refine: grew the red overlay. "
                shot = {"views": focus_view, "yaw": yaw, "pitch": pitch, "zoom": float(zoom or 1.0)}
            else:
                topo = _active_click_topo(ctx, src.name)
                if topo:
                    result = _build_click_mask(
                        ctx,
                        loaded,
                        src,
                        topo=topo,
                        target=hint,
                        focus_view=focus_view,
                        yaw=yaw,
                        pitch=pitch,
                        zoom=float(zoom or 1.0),
                    )
                else:
                    result = _build_auto_mask(
                        ctx,
                        loaded,
                        src,
                        target=hint,
                        focus_view=focus_view,
                        yaw=yaw,
                        pitch=pitch,
                        zoom=float(zoom or 1.0),
                    )
                mask = result["mask"]
                shot = dict(result["shot"])
                note = str(result.get("note") or "")
                verdict = dict(result.get("verdict") or {})
                _store_mask_state(
                    ctx,
                    src.name,
                    target=hint,
                    result=result,
                    proposal_status="ready" if verdict.get("verdict") == "ok" else "need_click",
                )
        except TopoError as exc:
            return f"mask_mesh missed: {exc} Change views/x/y or click the extra bit."
        except Exception as extra_exc:
            return f"mask_mesh failed: {extra_exc}"
        if not mask.any():
            return (
                f"mask_mesh failed to build a confident automatic proposal on {src.name}. "
                f"{note}"
                "The multi-view detector could not isolate the extra bit. "
                "Click the flap on a look PNG, then rerun mask_mesh. Do NOT call remove_mesh."
            )
        n = int(mask.sum())
        pct = 100.0 * n / max(len(loaded.faces), 1)
        metrics = mask_geometry_metrics(loaded, mask, verts=loaded.vertices, faces=loaded.faces)
        if mask_is_tiny(mask, loaded.vertices, loaded.faces):
            return (
                f"mask_mesh needs click: automatic proposal on {src.name} is too small ({n} faces). "
                f"{note}"
                "Red would be invisible, so the detector probably locked onto the wrong place. "
                "Click the extra bit on a look PNG, then rerun mask_mesh. "
                "Do NOT call remove_mesh."
            )
        failure = _auto_acceptance_failure(metrics, verdict)
        if failure:
            return (
                f"mask_mesh needs click: automatic proposal on {src.name} is not trustworthy ({failure}). "
                f"{note}"
                "The current red patch does not look like a whole extra protrusion. "
                "Click the extra bit on a look PNG, then rerun mask_mesh. Do NOT call remove_mesh."
            )
        if pct > 8:
            return (
                f"mask_mesh needs click: automatic proposal painted {n} faces ({pct:.0f}% of {src.name}). "
                f"{note}"
                "That still looks like the skirt/body, not the extra bit. "
                "Click the petal/spike on a look PNG, then rerun mask_mesh. Do NOT call remove_mesh."
            )
        review_verdict = str(verdict.get("verdict") or "")
        if action not in {"shrink", "grow"} and review_verdict not in {"", "ok"}:
            emit_masked_mesh_view(
                ctx,
                src,
                loaded,
                mask,
                camera=str(shot.get("views") or "right"),
                zoom=float(shot.get("zoom") or 1.5),
            )
            extra = _preview_mask(ctx, src, mask, shot)
            return (
                f"mask_mesh tentative proposal on {src.name}: {n} faces red. "
                f"{note}{extra} "
                f"Auto-review verdict is {review_verdict}, so this is NOT ready for remove_mesh. "
                "Ask the user to click the extra bit on a PNG, then rerun mask_mesh."
            )
        ctx.deps.store.set_mesh_mask(ctx.deps.chat_id, src.name, mask.nonzero()[0])
        ctx.deps.store.clear_mesh_target(ctx.deps.chat_id)
        emit_masked_mesh_view(
            ctx,
            src,
            loaded,
            mask,
            camera=str(shot.get("views") or "right"),
            zoom=float(shot.get("zoom") or 1.5),
        )
        extra = _preview_mask(ctx, src, mask, shot)
        verdict_txt = ""
        if action in {"shrink", "grow"}:
            verdict_txt = note
        else:
            verdict_txt = note + " " if note else ""
        painted = (
            f"Prepared automatic mask proposal: {n} faces red on {src.name}. "
            f"{verdict_txt}{extra}"
            "Red overlay = candidate extra bit. "
            "Open «маска · 3D» to rotate. "
            "The proposal was built from several cameras and auto-reviewed before showing it. "
            "Do NOT call remove_mesh until the user confirms this overlay."
        )
        if apply:
            return (
                painted
                + "apply=True ignored: show the red overlay first. "
                "If the user already confirmed this overlay, call remove_mesh."
            )
        return (
            painted
            + "If the user wants another visual check, you may call look(target='mesh'), but it is no longer required for the mask workflow."
        )


def _goal_hint(ctx: RunContext[ChatDeps], describe: str | None) -> str:
    hint = (describe or "").strip()
    if hint:
        return hint
    try:
        from mesh_forge.agent.workspace import _goal_from_messages, _safe_messages

        return _goal_from_messages(_safe_messages(ctx.deps.store, ctx.deps.chat_id))
    except Exception:
        return ""


def _active_click_topo(ctx: RunContext[ChatDeps], mesh_name: str) -> dict:
    topo = dict(ctx.deps.store.active_mesh_topo(ctx.deps.chat_id) or {})
    if int(topo.get("face", -1)) < 0 and int(topo.get("vertex", -1)) < 0:
        return {}
    topo_mesh = str(topo.get("mesh") or "").strip()
    if topo_mesh and topo_mesh != mesh_name:
        return {}
    return topo


def _auto_acceptance_failure(metrics: dict, verdict: dict) -> str:
    kind = str(verdict.get("verdict") or "").strip().lower()
    if kind in {"tiny_spot", "partial", "wrong", "missed", "too_much", "too_little"}:
        return kind.replace("_", " ")
    if bool(metrics.get("is_slab")):
        return "flat skirt patch"
    if float(metrics.get("outward_score") or -1.0) < 0.0:
        return "no protrusion geometry"
    if int(metrics.get("largest_component_faces") or 0) <= 3:
        return "single crumb component"
    if float(metrics.get("area_frac") or 0.0) < 0.0025:
        return "tiny local patch"
    return ""


def _resolve_focus_view(
    ctx: RunContext[ChatDeps],
    *,
    views: str | None,
    yaw: float | None,
    pitch: float | None,
    zoom: float,
    hint: str = "",
) -> tuple[str, float | None, float | None, float]:
    meta = ctx.deps.store.get_meta(ctx.deps.chat_id)
    look = dict(meta.look_view or {})
    cam = str(views or look.get("views") or "right")
    cam, yaw, pitch, zoom = inherit_look_camera(ctx, cam, yaw, pitch, zoom)
    named = str(cam or "right").split(",")[0].strip().lower() or "right"
    if named not in {"front", "left", "right", "back", "top", "viewer"}:
        named = "right"
    if not (views or "").strip() and named == "viewer":
        named = _semantic_focus_view(hint, named)
    return named, yaw, pitch, float(zoom or 1.0)


def _semantic_focus_view(hint: str, current: str = "right") -> str:
    text = (hint or "").lower()
    if any(word in text for word in ("сзади", "зад", "back")):
        return "back"
    if any(word in text for word in ("спереди", "перед", "front")):
        return "front"
    if any(word in text for word in ("сверху", "верх", "top")):
        return "top"
    side = mask_aim_side(hint, None)
    if side in {"left", "right"}:
        return side
    return (current or "right").strip().lower() or "right"


def _mask_log(ctx: RunContext[ChatDeps], text: str) -> None:
    line = (text or "").strip()
    if line:
        ctx.deps.emit_event("tool_text_delta", delta=f"{line}\n")


def _mask_think(ctx: RunContext[ChatDeps], text: str) -> None:
    line = (text or "").strip()
    if line:
        ctx.deps.emit_event("tool_thinking_delta", delta=f"{line}\n")


def _emit_mask_preview(ctx: RunContext[ChatDeps], path: Path, *, label: str, view: str) -> None:
    src = Path(path)
    if not src.is_file():
        return
    files = ctx.deps.store.files_dir(ctx.deps.chat_id).resolve()
    dest = src
    if src.resolve().parent != files:
        dest = ctx.deps.store.new_file(
            ctx.deps.chat_id,
            f"{label}_{view}{src.suffix or '.png'}",
        )
        dest.write_bytes(src.read_bytes())
    art = ctx.deps.store.artifact_from_path(ctx.deps.chat_id, dest, label=label, view=view)
    ctx.deps.emit_artifact(art)


def _mask_view_pack(focus_view: str) -> list[str]:
    focus = (focus_view or "right").strip().lower() or "right"
    packs = {
        "right": ["right", "front", "back"],
        "left": ["left", "front", "back"],
        "front": ["front", "right", "left"],
        "back": ["back", "right", "left"],
        "top": ["top", "front", "right"],
        "viewer": ["viewer", "right", "front"],
    }
    return list(packs.get(focus, packs["right"]))


def _review_view_pack(focus_view: str) -> list[str]:
    focus = (focus_view or "right").strip().lower() or "right"
    return [focus]


def _score_observations(focus_view: str, observations: list[dict], *, limit: int = 2) -> list[dict]:
    focus = (focus_view or "right").strip().lower() or "right"
    preferred = _mask_view_pack(focus)
    order = {name: index for index, name in enumerate(preferred)}
    ranked = sorted(
        (dict(obs) for obs in observations if obs.get("visible")),
        key=lambda obs: (
            0 if str(obs.get("view") or "").strip().lower() == focus else 1,
            order.get(str(obs.get("view") or "").strip().lower(), 99),
            -float(obs.get("confidence") or 0.0),
        ),
    )
    if ranked:
        return ranked[: max(1, int(limit))]
    return [dict(obs) for obs in observations[: max(1, int(limit))]]


def _render_detection_pack(
    ctx: RunContext[ChatDeps],
    src: Path,
    mesh,
    *,
    focus_view: str,
    zoom: float,
    yaw: float | None,
    pitch: float | None,
) -> list[dict]:
    from mesh_forge.render import _camera_eye_target, _seat_for_viewer, render_mesh_preview

    verts, faces, extent = _seat_for_viewer(mesh)
    _ = faces
    records: list[dict] = []
    cameras = _mask_view_pack(focus_view)
    _mask_log(ctx, f"Mask detect views: {', '.join(cameras)}.")
    for camera in cameras:
        shot_zoom = float(max(zoom, 1.2) if camera == focus_view else 1.05)
        shot_yaw = yaw if camera == focus_view else None
        shot_pitch = pitch if camera == focus_view else None
        preview = ctx.deps.store.new_file(ctx.deps.chat_id, f"mask_detect_{camera}.png")
        render_mesh_preview(
            src,
            preview,
            camera=camera,
            zoom=shot_zoom,
            mesh=mesh,
            yaw=shot_yaw,
            pitch=shot_pitch,
        )
        _emit_mask_preview(ctx, preview, label=f"маска · detect {camera}", view=camera)
        eye, target = _camera_eye_target(
            extent,
            camera,
            pad=1.0,
            zoom=shot_zoom,
            yaw=shot_yaw,
            pitch=shot_pitch,
        )
        records.append(
            {
                "view": camera,
                "path": preview,
                "eye": eye,
                "target": target,
                "zoom": shot_zoom,
            }
        )
    return records


def _detect_multi_view(ctx: RunContext[ChatDeps], target: str, records: list[dict]) -> list[dict]:
    from mesh_forge.backends.lmstudio import LMStudioClient

    images = [(str(rec["view"]), rec["path"]) for rec in records]
    observations = LMStudioClient().detect_mesh_part_multi_view(images, target=target)
    by_view = {str(obs.get("view") or ""): obs for obs in observations}
    out: list[dict] = []
    for rec in records:
        obs = by_view.get(str(rec["view"]))
        if not obs:
            continue
        out.append({**obs, "eye": rec["eye"], "target": rec["target"], "zoom": rec["zoom"]})
    if out:
        summary = "; ".join(
            f"{obs.get('view')}={'yes' if obs.get('visible') else 'no'} conf={float(obs.get('confidence') or 0.0):.2f}"
            for obs in out
        )
        _mask_think(ctx, f"Vision observations: {summary}.")
    else:
        _mask_think(ctx, "Vision observations: no usable boxes returned.")
    return out


def _mask_bbox_from_preview(path: Path) -> dict[str, float] | None:
    from PIL import Image

    image = Image.open(path).convert("L")
    arr = np.asarray(image, dtype=np.uint8)
    if arr.size == 0 or int(arr.max()) < 16:
        return None
    threshold = max(16, int(arr.max() * 0.35))
    ys, xs = np.nonzero(arr >= threshold)
    if xs.size == 0 or ys.size == 0:
        return None
    h, w = arr.shape
    x0 = float(xs.min()) / max(1.0, float(w - 1))
    x1 = float(xs.max()) / max(1.0, float(w - 1))
    y0 = float(ys.min()) / max(1.0, float(h - 1))
    y1 = float(ys.max()) / max(1.0, float(h - 1))
    if x1 - x0 < 0.002 or y1 - y0 < 0.002:
        return None
    return {"x0": x0, "y0": y0, "x1": x1, "y1": y1}


def _bbox_is_too_broad(bbox: dict[str, float]) -> bool:
    width = float(bbox["x1"]) - float(bbox["x0"])
    height = float(bbox["y1"]) - float(bbox["y0"])
    area = width * height
    return area >= 0.08 or height >= 0.50 or width >= 0.55


def _detect_multi_view_with_comfy(
    ctx: RunContext[ChatDeps],
    target: str,
    records: list[dict],
) -> list[dict]:
    from mesh_forge.adapters.comfyui_client import ComfyUiClient
    from mesh_forge.config import load_config

    try:
        seg_cfg = load_config().segmentation
    except Exception:
        return []
    if not bool(seg_cfg.enabled):
        return []

    client = ComfyUiClient()
    out: list[dict] = []
    views = records[:1]
    _mask_log(
        ctx,
        f"Comfy segmentation enabled: text-masking {len(views)} views before VLM boxes.",
    )
    for index, rec in enumerate(views, start=1):
        view = str(rec.get("view") or "right")
        image_path = Path(rec["path"])
        _mask_log(ctx, f"Comfy segmentation {index}/{len(views)} on {view}.")
        try:
            result = client.segment_view_by_text(
                target,
                image_path,
                ctx.deps.files_dir() / "segmentation",
                project_id=ctx.deps.chat_id,
                max_detections=1,
                confidence_threshold=float(seg_cfg.mask_threshold or 0.2),
            )
        except Exception as exc:
            _mask_think(ctx, f"Comfy segmentation on {view} failed: {exc}.")
            continue
        _emit_mask_preview(
            ctx,
            result.visualization.path,
            label=f"маска · comfy {view}",
            view=view,
        )
        _emit_mask_preview(
            ctx,
            result.mask.path,
            label=f"маска · comfy mask {view}",
            view=view,
        )
        bbox = dict(result.boxes[0]) if result.boxes else _mask_bbox_from_preview(result.mask.path)
        if not bbox:
            _mask_think(ctx, f"Comfy segmentation on {view} returned no usable mask bbox.")
            continue
        area = float((bbox["x1"] - bbox["x0"]) * (bbox["y1"] - bbox["y0"]))
        if _bbox_is_too_broad(bbox):
            _mask_think(
                ctx,
                f"Comfy segmentation on {view}: bbox too broad (area={area:.3f}); ignoring this view.",
            )
            continue
        score = float(result.scores[0]) if result.scores else max(0.25, min(0.95, area * 4.0 + 0.25))
        source = "sam3 boxes" if result.boxes else "mask preview"
        _mask_think(ctx, f"Comfy segmentation on {view}: bbox area={area:.3f} via {source}.")
        out.append(
            {
                "view": view,
                "visible": True,
                "confidence": max(0.25, min(0.95, score if score <= 1.0 else score / 100.0)),
                "kind": "sam3_grounding",
                "touchesBody": True,
                "note": f"bbox derived from comfy {source}",
                "eye": rec["eye"],
                "target": rec["target"],
                "zoom": rec["zoom"],
                **bbox,
            }
        )
    if out:
        summary = "; ".join(
            f"{obs.get('view')} bbox=({float(obs['x0']):.2f},{float(obs['y0']):.2f})-({float(obs['x1']):.2f},{float(obs['y1']):.2f})"
            for obs in out
        )
        _mask_think(ctx, f"Comfy observations: {summary}.")
    else:
        _mask_think(ctx, "Comfy segmentation produced no usable observations; falling back to VLM box detect.")
    return out


def _build_auto_mask(
    ctx: RunContext[ChatDeps],
    mesh,
    src: Path,
    *,
    target: str,
    focus_view: str,
    yaw: float | None,
    pitch: float | None,
    zoom: float,
    review: bool = True,
) -> dict:
    from mesh_forge.render import _seat_for_viewer

    if not (target or "").strip():
        raise TopoError("Need describe=... so vision knows which extra bit to isolate.")
    verts, faces, _extent = _seat_for_viewer(mesh)
    _mask_log(ctx, f"Building automatic mask for: {target}. Focus view: {focus_view}.")
    best: dict | None = None
    best_score = -1.0
    min_views = 2
    try:
        from mesh_forge.config import load_config

        min_views = max(1, int(load_config().segmentation.projection_min_views or 2))
    except Exception:
        pass
    passes = [float(max(zoom, 1.0)), float(max(1.55, zoom * 1.35))]
    for index, detect_zoom in enumerate(passes):
        _mask_log(ctx, f"Detect pass {index + 1}: render + detect at zoom {detect_zoom:.2f}.")
        records = _render_detection_pack(
            ctx,
            src,
            mesh,
            focus_view=focus_view,
            zoom=detect_zoom,
            yaw=yaw,
            pitch=pitch,
        )
        observations = _detect_multi_view_with_comfy(ctx, target, records)
        if not observations:
            observations = _detect_multi_view(ctx, target, records)
        if not observations:
            continue
        view_limit = max(min_views, len(observations))
        scored_observations = _score_observations(focus_view, observations, limit=view_limit)
        _mask_log(
            ctx,
            "Scoring candidate from views: "
            + ", ".join(str(obs.get("view") or "") for obs in scored_observations)
            + ".",
        )
        n_faces = int(len(faces))
        if n_faces > 80_000:
            _mask_log(
                ctx,
                f"Skipping slow 3D box projection on {n_faces} faces; use a click or the geometry cut instead.",
            )
            continue
        _mask_log(ctx, f"Projecting 2D boxes onto {n_faces} faces.")
        mask, scores = mask_from_view_observations(mesh, verts, faces, scored_observations)
        score_sum = float(np.maximum(scores, 0.0).sum())
        _mask_think(ctx, f"Candidate size after scoring: {int(mask.sum())} faces, score_sum={score_sum:.1f}.")
        if score_sum > best_score:
            best_score = score_sum
            best = {
                "mask": mask,
                "scores": scores,
                "observations": scored_observations,
                "detect_zoom": detect_zoom,
            }
        visible = [obs for obs in scored_observations if obs.get("visible")]
        # A confident first pass should not pay for a second full detect pack.
        if index == 0 and np.any(mask) and len(visible) >= min_views and score_sum > 0.0:
            break
    if best is None:
        return {
            "mask": np.zeros(int(len(mesh.faces)), dtype=bool),
            "shot": {"views": focus_view, "yaw": yaw, "pitch": pitch, "zoom": float(zoom or 1.0)},
            "note": "Vision did not return usable multi-view observations.",
            "verdict": {},
            "detect_views": [],
            "review_views": [],
        }
    mask = np.asarray(best["mask"], dtype=bool)
    verdict: dict = {}
    review_views: list[str] = []
    if review and np.any(mask):
        _mask_log(ctx, f"Reviewing candidate mask of {int(mask.sum())} faces.")
        verdict, review_views = _review_mask_candidate(
            ctx,
            src,
            mesh,
            mask,
            target=target,
            focus_view=focus_view,
            zoom=float(best["detect_zoom"]),
        )
        if verdict:
            _mask_think(
                ctx,
                f"Review verdict: {str(verdict.get('verdict') or '')} (conf={float(verdict.get('confidence') or 0.0):.2f}).",
            )
    note_bits = []
    visible = [obs for obs in best["observations"] if obs.get("visible")]
    if visible:
        note_bits.append(
            f"Multi-view detect: {len(visible)}/{len(best['observations'])} cameras saw the target."
        )
    if verdict.get("note"):
        note_bits.append(str(verdict["note"]).strip())
    return {
        "mask": mask,
        "shot": {
            "views": focus_view,
            "yaw": yaw,
            "pitch": pitch,
            "zoom": float(max(best["detect_zoom"], 1.35)),
        },
        "note": " ".join(bit for bit in note_bits if bit).strip(),
        "verdict": verdict,
        "detect_views": [str(obs.get("view") or "") for obs in best["observations"]],
        "observations": [
            {
                "view": str(obs.get("view") or ""),
                "visible": bool(obs.get("visible")),
                "confidence": float(obs.get("confidence") or 0.0),
                "kind": str(obs.get("kind") or ""),
                "touchesBody": bool(obs.get("touchesBody")),
                "note": str(obs.get("note") or ""),
                **(
                    {
                        "x0": float(obs["x0"]),
                        "y0": float(obs["y0"]),
                        "x1": float(obs["x1"]),
                        "y1": float(obs["y1"]),
                    }
                    if obs.get("visible")
                    else {}
                ),
            }
            for obs in best["observations"]
        ],
        "review_views": review_views,
    }


def _build_click_mask(
    ctx: RunContext[ChatDeps],
    mesh,
    src: Path,
    *,
    topo: dict,
    target: str,
    focus_view: str,
    yaw: float | None,
    pitch: float | None,
    zoom: float,
) -> dict:
    from mesh_forge.render import _camera_eye_target, _seat_for_viewer

    verts, faces, extent = _seat_for_viewer(mesh)
    seed_face = int(topo.get("face", -1))
    _mask_log(ctx, f"Using user click/topology as seed: face={seed_face}.")
    if seed_face < 0:
        mask = face_mask_for_topo(mesh, topo)
    else:
        eye, cam_target = _camera_eye_target(
            extent,
            focus_view,
            pad=1.0,
            zoom=float(max(zoom, 1.2)),
            yaw=yaw,
            pitch=pitch,
        )
        mask = grow_visible_lump(
            mesh,
            seed_face,
            seated_verts=verts,
            seated_faces=faces,
            eye=eye,
            target=cam_target,
        )
        if not np.any(mask):
            mask = face_mask_for_topo(mesh, topo)
    verdict: dict = {}
    review_views: list[str] = []
    if np.any(mask):
        _mask_think(ctx, f"Click-seeded candidate size: {int(mask.sum())} faces.")
        verdict, review_views = _review_mask_candidate(
            ctx,
            src,
            mesh,
            mask,
            target=target or "clicked extra bit",
            focus_view=focus_view,
            zoom=float(max(zoom, 1.2)),
        )
    note = "Click-seeded proposal from the selected mesh face."
    if verdict.get("note"):
        note = f"{note} {str(verdict.get('note') or '').strip()}".strip()
    return {
        "mask": np.asarray(mask, dtype=bool),
        "shot": {
            "views": focus_view,
            "yaw": yaw,
            "pitch": pitch,
            "zoom": float(max(zoom, 1.2)),
        },
        "note": note,
        "verdict": verdict,
        "detect_views": ["click"],
        "observations": [
            {
                "view": "click",
                "visible": True,
                "confidence": 1.0,
                "kind": "seed",
                "touchesBody": True,
                "note": "user clicked target",
            }
        ],
        "review_views": review_views,
    }


def _review_mask_candidate(
    ctx: RunContext[ChatDeps],
    src: Path,
    mesh,
    mask: np.ndarray,
    *,
    target: str,
    focus_view: str,
    zoom: float,
) -> tuple[dict, list[str]]:
    from mesh_forge.backends.lmstudio import LMStudioClient
    from mesh_forge.render import render_mesh_preview

    review_views = _review_view_pack(focus_view)
    _mask_log(ctx, f"Review views: {', '.join(review_views)}.")
    images: list[tuple[str, Path]] = []
    for camera in review_views:
        preview = ctx.deps.store.new_file(ctx.deps.chat_id, f"mask_review_{camera}.png")
        render_mesh_preview(
            src,
            preview,
            camera=camera,
            zoom=max(float(zoom or 1.0), 1.25 if camera == focus_view else 1.0),
            mesh=mesh,
            face_mask=mask,
        )
        _emit_mask_preview(ctx, preview, label=f"маска · review {camera}", view=camera)
        images.append((camera, preview))
    verdict = LMStudioClient().review_mesh_mask(images, target=target) or {}
    return verdict, review_views


def _apply_review_refinement(
    mesh,
    mask: np.ndarray,
    verdict: str,
    *,
    verts: np.ndarray,
    faces: np.ndarray,
) -> np.ndarray:
    current = np.asarray(mask, dtype=bool)
    if verdict == "too_much":
        return erode_face_mask(mesh, current, hops=2)
    if verdict == "too_little":
        return dilate_face_mask(mesh, current, hops=2, seated_verts=verts, seated_faces=faces)
    return current


def _store_mask_state(
    ctx: RunContext[ChatDeps],
    mesh_name: str,
    *,
    target: str,
    result: dict,
    proposal_status: str,
) -> None:
    verdict = dict(result.get("verdict") or {})
    ctx.deps.store.set_mask_state(
        ctx.deps.chat_id,
        {
            "tool": "mask_mesh",
            "mesh": mesh_name,
            "target": target,
            "proposal_status": proposal_status,
            "detect_views": list(result.get("detect_views") or []),
            "observations": list(result.get("observations") or []),
            "review_views": list(result.get("review_views") or []),
            "candidate_faces": int(np.asarray(result.get("mask"), dtype=bool).sum()),
            "confidence": float(verdict.get("confidence") or 0.0),
            "needs_click": proposal_status != "ready",
            "review_verdict": str(verdict.get("verdict") or ""),
            "note": str(result.get("note") or ""),
        },
    )


def _preview_mask(ctx: RunContext[ChatDeps], src, mask, shot: dict) -> str:
    extra = ""
    try:
        from mesh_forge.tools.look import _render_mesh_looks

        cam = str(shot.get("views") or "right").split(",")[0].strip() or "right"
        _render_mesh_looks(
            ctx,
            src,
            views=cam,
            zoom=max(float(shot.get("zoom") or 1.0), 1.3),
            region="",
            pick=None,
            yaw=shot.get("yaw"),
            pitch=shot.get("pitch"),
            face_mask=mask,
            mask_overlay=True,
            emit=True,
        )
    except Exception:
        extra += " Red overlay skipped."
    return extra
