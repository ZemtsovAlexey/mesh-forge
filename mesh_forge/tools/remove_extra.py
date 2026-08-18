from __future__ import annotations

import json
import re

import numpy as np
from pydantic_ai import RunContext

from mesh_forge.agent.deps import ChatDeps
from mesh_forge.backends.lmstudio import LMStudioClient
from mesh_forge.ops.edit import EditError
from mesh_forge.ops.geometry import load_mesh
from mesh_forge.ops.remove import build_auto_remove_proposal, classify_removal_strategy
from mesh_forge.ops.topo import dilate_face_mask
from mesh_forge.tools.base import MeshTool
from mesh_forge.tools.common import (
    LOOK_AFTER,
    apply_saved_proposal_mesh,
    carve_painted_mask,
    emit_masked_mesh_view,
    resolve_mesh,
    save_mesh_artifact,
)
from mesh_forge.tools.mask_mesh import MaskMesh, _goal_hint, _review_view_pack

_REJECTION = re.compile(
    r"^(нет+|неа|не надо|не нужно|не то|неправильно|мимо|no+|nope|wrong|not this)\b[\s!.…]*.*$",
    re.IGNORECASE,
)


class RemoveExtra(MeshTool):
    title = "Удалить лишнее"
    heavy = True

    def run(
        self,
        ctx: RunContext[ChatDeps],
        describe: str | None = None,
        mesh_ref: str | None = None,
        apply: bool = False,
        views: str | None = None,
        x: float | None = None,
        y: float | None = None,
        yaw: float | None = None,
        pitch: float | None = None,
        zoom: float = 1.0,
    ) -> str:
        """Universal delete entrypoint: choose strategy, preview proposal, then apply on confirmation."""
        src = resolve_mesh(ctx, mesh_ref)
        if apply:
            state = ctx.deps.store.removal_state(ctx.deps.chat_id)
            if not state:
                return "remove_extra apply skipped: no prepared proposal. Build a proposal first."
            if _latest_user_rejected(ctx):
                ctx.deps.store.clear_removal_state(ctx.deps.chat_id)
                return (
                    "remove_extra apply blocked: the latest user message rejected this proposal. "
                    "Build a new proposal instead of applying the old one."
                )
            wanted = str(state.get("mesh") or "")
            if wanted and wanted != src.name:
                return "remove_extra apply skipped: proposal belongs to another mesh. Rebuild the proposal."
            art = apply_saved_proposal_mesh(
                ctx,
                state,
                filename="removed_extra.stl",
                label="removed",
                role="edit",
            )
            ctx.deps.store.clear_removal_state(ctx.deps.chat_id)
            ctx.deps.store.clear_mask_state(ctx.deps.chat_id)
            ctx.deps.store.clear_mesh_target(ctx.deps.chat_id)
            return (
                f"Applied {state.get('strategy') or 'remove'} proposal on {src.name} → {art.name}. "
                + LOOK_AFTER
            )

        hint = _goal_hint(ctx, describe)
        if not hint:
            return "remove_extra needs describe=... so it can choose a removal strategy."
        ctx.deps.store.clear_removal_state(ctx.deps.chat_id)
        strategy = classify_removal_strategy(hint)
        if strategy == "surface_patch":
            return self._run_surface_patch(
                ctx,
                src,
                hint,
                views=views,
                x=x,
                y=y,
                yaw=yaw,
                pitch=pitch,
                zoom=zoom,
            )

        loaded = load_mesh(src)
        painted = _segmentation_mask_candidate(
            ctx,
            src,
            loaded,
            hint,
            views=views,
            yaw=yaw,
            pitch=pitch,
            zoom=zoom,
        )
        try:
            result = build_auto_remove_proposal(loaded, hint)
        except EditError as exc:
            if painted is not None:
                return self._finish_mask_proposal(
                    ctx,
                    src,
                    loaded,
                    painted,
                    hint,
                    strategy="segmentation",
                    views=views,
                    zoom=zoom,
                    prefix=f"{strategy} fallback via SAM3: {exc}. ",
                )
            if strategy != "surface_patch":
                return self._run_surface_patch(
                    ctx,
                    src,
                    hint,
                    views=views,
                    x=x,
                    y=y,
                    yaw=yaw,
                    pitch=pitch,
                    zoom=zoom,
                    prefix=f"{strategy} fallback: {exc}. ",
                )
            return f"remove_extra failed: {exc}"
        if painted is not None:
            candidates = [painted, *list(result.get("candidate_masks") or [])]
            result["candidate_masks"] = candidates
            _remove_log(ctx, f"Added SAM3 mask candidate with {int(painted.sum())} faces.")
        if result.get("candidate_masks"):
            _remove_log(
                ctx,
                f"Candidates before VLM rerank: {len(result.get('candidate_masks') or [])}.",
            )
            reranked = _rerank_protrusion_candidates(
                ctx,
                src,
                loaded,
                result,
                target=hint,
                focus_view=str(views or "right"),
                zoom=float(zoom or 1.0),
            )
            if reranked is not None:
                result = reranked
        mask = np.asarray(result.get("mask"), dtype=bool)
        if mask.shape[0] == len(loaded.faces) and np.any(mask):
            emit_masked_mesh_view(
                ctx,
                src,
                loaded,
                mask,
                camera=str(views or "right"),
                zoom=max(float(zoom or 1.0), 1.3),
            )
        art = save_mesh_artifact(
            ctx,
            result["mesh"],
            f"{result['strategy']}_proposal.stl",
            label=f"proposal {result['strategy']}",
            role="edit",
            make_current=False,
        )
        ctx.deps.store.set_removal_state(
            ctx.deps.chat_id,
            {
                "tool": "remove_extra",
                "mesh": src.name,
                "target": hint,
                "strategy": str(result.get("strategy") or strategy),
                "proposal_status": "ready",
                "proposal_mesh": art.name,
                "candidate_faces": int(mask.sum()) if mask.size else int(result.get("stats", {}).get("faces_dropped") or 0),
                "note": str(result.get("note") or ""),
                "candidate_meta": list(result.get("debug_candidates") or []),
                "chosen_candidate_verdict": str(result.get("chosen_candidate_verdict") or ""),
                "chosen_candidate_score": float(result.get("chosen_candidate_score") or 0.0),
            },
        )
        _emit_remove_debug_report(
            ctx,
            {
                "mesh": src.name,
                "target": hint,
                "strategy": str(result.get("strategy") or strategy),
                "candidate_faces": int(mask.sum()) if mask.size else int(result.get("stats", {}).get("faces_dropped") or 0),
                "note": str(result.get("note") or ""),
                "candidate_meta": list(result.get("debug_candidates") or []),
                "chosen_candidate_verdict": str(result.get("chosen_candidate_verdict") or ""),
                "chosen_candidate_score": float(result.get("chosen_candidate_score") or 0.0),
            },
        )
        ctx.deps.store.clear_mesh_target(ctx.deps.chat_id)
        return (
            f"Prepared removal proposal with strategy {result.get('strategy')}: {result.get('note') or ''} "
            f"Open the proposal preview and compare it to the request. "
            "If the user confirms, call remove_extra(apply=True). "
            "If this proposal looks wrong, you may still fall back to mask_mesh for a surface patch."
        )

    def _run_surface_patch(
        self,
        ctx: RunContext[ChatDeps],
        src,
        hint: str,
        *,
        views: str | None,
        x: float | None,
        y: float | None,
        yaw: float | None,
        pitch: float | None,
        zoom: float,
        prefix: str = "",
    ) -> str:
        note = MaskMesh().run(
            ctx,
            views=views,
            x=x,
            y=y,
            yaw=yaw,
            pitch=pitch,
            zoom=zoom,
            mesh_ref=src.name,
            describe=hint,
            apply=False,
        )
        loaded = load_mesh(src)
        try:
            out_mesh, stats = carve_painted_mask(ctx, loaded, src.name)
        except Exception:
            ctx.deps.store.clear_removal_state(ctx.deps.chat_id)
            return prefix + note
        art = save_mesh_artifact(
            ctx,
            out_mesh,
            "surface_patch_proposal.stl",
            label="proposal surface_patch",
            role="edit",
            make_current=False,
        )
        ctx.deps.store.set_removal_state(
            ctx.deps.chat_id,
            {
                "tool": "remove_extra",
                "mesh": src.name,
                "target": hint,
                "strategy": "surface_patch",
                "proposal_status": "ready",
                "proposal_mesh": art.name,
                "candidate_faces": int(stats.get("faces_dropped") or 0),
                "note": "Surface patch proposal from mask_mesh.",
            },
        )
        return (
            prefix
            + "Prepared surface-patch removal proposal via mask workflow. "
            "Check the red overlay and the proposal preview. "
            "If the user confirms, call remove_extra(apply=True). "
            + note
        )

    def _finish_mask_proposal(
        self,
        ctx: RunContext[ChatDeps],
        src,
        loaded,
        mask: np.ndarray,
        hint: str,
        *,
        strategy: str,
        views: str | None,
        zoom: float,
        prefix: str = "",
    ) -> str:
        emit_masked_mesh_view(
            ctx,
            src,
            loaded,
            mask,
            camera=str(views or "right"),
            zoom=max(float(zoom or 1.0), 1.3),
        )
        from mesh_forge.ops.geometry import carve_faces

        out_mesh, stats = carve_faces(loaded, mask, min_keep_ratio=0.50, min_keep_faces=8, drop_crumbs=False)
        art = save_mesh_artifact(
            ctx,
            out_mesh,
            f"{strategy}_proposal.stl",
            label=f"proposal {strategy}",
            role="edit",
            make_current=False,
        )
        ctx.deps.store.set_removal_state(
            ctx.deps.chat_id,
            {
                "tool": "remove_extra",
                "mesh": src.name,
                "target": hint,
                "strategy": strategy,
                "proposal_status": "ready",
                "proposal_mesh": art.name,
                "candidate_faces": int(mask.sum()),
                "note": "Proposal from Comfy SAM3 multi-view mask.",
            },
        )
        ctx.deps.store.clear_mesh_target(ctx.deps.chat_id)
        return (
            prefix
            + f"Prepared removal proposal with strategy {strategy} from Comfy segmentation "
            f"({int(mask.sum())} faces). Open the proposal preview and compare it to the request. "
            "If the user confirms, call remove_extra(apply=True)."
        )


def _segmentation_mask_candidate(
    ctx: RunContext[ChatDeps],
    src,
    loaded,
    hint: str,
    *,
    views: str | None,
    yaw: float | None,
    pitch: float | None,
    zoom: float,
):
    try:
        from mesh_forge.config import load_config

        if not bool(load_config().segmentation.enabled):
            return None
    except Exception:
        return None
    from mesh_forge.tools.mask_mesh import _build_auto_mask

    _remove_log(ctx, "Segmentation-first: building a SAM3 multi-view mask before geometry cut.")
    try:
        result = _build_auto_mask(
            ctx,
            loaded,
            src,
            target=hint,
            focus_view=str(views or "right"),
            yaw=yaw,
            pitch=pitch,
            zoom=float(zoom or 1.0),
            review=False,
        )
    except Exception as exc:
        _remove_log(ctx, f"Segmentation-first failed: {exc}.")
        return None
    mask = np.asarray(result.get("mask"), dtype=bool)
    if mask.shape[0] != len(loaded.faces) or not np.any(mask):
        _remove_log(ctx, "Segmentation-first produced an empty 3D mask.")
        return None
    _remove_log(ctx, f"Segmentation-first painted {int(mask.sum())} faces.")
    return mask


def _latest_user_rejected(ctx: RunContext[ChatDeps]) -> bool:
    try:
        messages = ctx.deps.store.load_messages(ctx.deps.chat_id)
    except Exception:
        return False
    for message in reversed(messages):
        if message.role != "user":
            continue
        text = (message.content or "").strip()
        return bool(_REJECTION.match(text))
    return False


def _rerank_protrusion_candidates(
    ctx: RunContext[ChatDeps],
    src,
    mesh,
    result: dict,
    *,
    target: str,
    focus_view: str,
    zoom: float,
) -> dict | None:
    candidates = [np.asarray(mask, dtype=bool) for mask in (result.get("candidate_masks") or [])]
    unique: list[np.ndarray] = []
    seen: set[bytes] = set()
    diagnostics: list[dict[str, object]] = []
    for mask in candidates:
        key = mask.tobytes()
        if np.any(mask) and key not in seen:
            unique.append(mask)
            seen.add(key)
    if not unique and np.any(geometry_mask := np.asarray(result.get("mask"), dtype=bool)):
        unique = [geometry_mask]
    geometry_mask = np.asarray(result.get("mask"), dtype=bool)
    best_mask = geometry_mask
    best_verdict = ""
    best_score = _candidate_review_score("", 0.0)
    if len(unique) <= 1:
        if unique:
            verdict, score = _review_mask_candidate_for_remove_extra(
                ctx,
                src,
                mesh,
                unique[0],
                target=target,
                focus_view=focus_view,
                zoom=zoom,
                tag="candidate_geometry",
            )
            _remove_log(ctx, f"Only candidate: verdict={verdict or 'none'} score={score:.2f}.")
            diagnostics.append(
                {
                    "tag": "candidate_geometry",
                    "faces": int(np.asarray(unique[0], dtype=bool).sum()),
                    "verdict": verdict,
                    "score": score,
                }
            )
            best_mask = np.asarray(unique[0], dtype=bool)
            best_verdict = verdict
            best_score = score
        else:
            _remove_log(ctx, "No protrusion candidates were generated.")
            result["debug_candidates"] = diagnostics
            result["chosen_candidate_verdict"] = ""
            result["chosen_candidate_score"] = 0.0
            return None
    reviewed: list[tuple[np.ndarray, str, float]] = []
    if len(unique) > 1:
        for idx, mask in enumerate(unique[:3]):
            verdict, score = _review_mask_candidate_for_remove_extra(
                ctx,
                src,
                mesh,
                mask,
                target=target,
                focus_view=focus_view,
                zoom=zoom,
                tag=f"candidate_{idx}",
            )
            _remove_log(ctx, f"Candidate {idx + 1}: verdict={verdict or 'none'} score={score:.2f}.")
            diagnostics.append(
                {
                    "tag": f"candidate_{idx}",
                    "faces": int(np.asarray(mask, dtype=bool).sum()),
                    "verdict": verdict,
                    "score": score,
                }
            )
            reviewed.append((mask, verdict, score))
            if score > best_score:
                best_score = score
                best_mask = mask
                best_verdict = verdict
    if best_verdict in {"partial", "too_little", "tiny_spot", "missed"} and len(reviewed) > 1:
        for idx, (mask, _verdict, _score) in enumerate(reviewed):
            if np.array_equal(mask, best_mask):
                continue
            union = np.asarray(best_mask | mask, dtype=bool)
            verdict, score = _review_mask_candidate_for_remove_extra(
                ctx,
                src,
                mesh,
                union,
                target=target,
                focus_view=focus_view,
                zoom=zoom,
                tag=f"candidate_union_{idx}",
            )
            _remove_log(ctx, f"Union candidate {idx + 1}: verdict={verdict or 'none'} score={score:.2f}.")
            diagnostics.append(
                {
                    "tag": f"candidate_union_{idx}",
                    "faces": int(np.asarray(union, dtype=bool).sum()),
                    "verdict": verdict,
                    "score": score,
                }
            )
            if score > best_score:
                best_score = score
                best_mask = union
                best_verdict = verdict
        if best_verdict in {"partial", "too_little", "tiny_spot", "missed"} and len(reviewed) > 2:
            union_all = np.zeros_like(best_mask, dtype=bool)
            for mask, _verdict, _score in reviewed:
                union_all |= mask
            verdict, score = _review_mask_candidate_for_remove_extra(
                ctx,
                src,
                mesh,
                union_all,
                target=target,
                focus_view=focus_view,
                zoom=zoom,
                tag="candidate_union_all",
            )
            _remove_log(ctx, f"Union all candidates: verdict={verdict or 'none'} score={score:.2f}.")
            diagnostics.append(
                {
                    "tag": "candidate_union_all",
                    "faces": int(np.asarray(union_all, dtype=bool).sum()),
                    "verdict": verdict,
                    "score": score,
                }
            )
            if score > best_score:
                best_score = score
                best_mask = union_all
                best_verdict = verdict
    if best_verdict in {"partial", "too_little", "tiny_spot", "missed"}:
        current = np.asarray(best_mask, dtype=bool)
        for step in range(1, 4):
            expanded = np.asarray(dilate_face_mask(mesh, current, hops=1), dtype=bool)
            if np.array_equal(expanded, current) or not np.any(expanded):
                break
            verdict, score = _review_mask_candidate_for_remove_extra(
                ctx,
                src,
                mesh,
                expanded,
                target=target,
                focus_view=focus_view,
                zoom=zoom,
                tag=f"candidate_expand_{step}",
            )
            _remove_log(ctx, f"Expanded candidate step {step}: verdict={verdict or 'none'} score={score:.2f}.")
            diagnostics.append(
                {
                    "tag": f"candidate_expand_{step}",
                    "faces": int(np.asarray(expanded, dtype=bool).sum()),
                    "verdict": verdict,
                    "score": score,
                }
            )
            if score > best_score:
                best_score = score
                best_mask = expanded
                best_verdict = verdict
            current = expanded
            if verdict == "ok":
                break
            if verdict == "too_much":
                break
    _remove_log(ctx, f"Chosen protrusion candidate verdict={best_verdict or 'none'} score={best_score:.2f}.")
    result["debug_candidates"] = diagnostics
    result["chosen_candidate_verdict"] = best_verdict
    result["chosen_candidate_score"] = best_score
    if np.array_equal(best_mask, geometry_mask):
        return None
    from mesh_forge.ops.geometry import carve_faces

    out_mesh, stats = carve_faces(mesh, best_mask, min_keep_ratio=0.50, min_keep_faces=8, drop_crumbs=False)
    updated = dict(result)
    updated["mask"] = np.asarray(best_mask, dtype=bool)
    updated["mesh"] = out_mesh
    updated["stats"] = stats
    updated["note"] = str(result.get("note") or "") + " VLM rerank/refine picked the best candidate."
    return updated


def _review_mask_candidate_for_remove_extra(
    ctx: RunContext[ChatDeps],
    src,
    mesh,
    mask: np.ndarray,
    *,
    target: str,
    focus_view: str,
    zoom: float,
    tag: str,
) -> tuple[str, float]:
    review_views = _review_view_pack(focus_view)
    images = []
    for camera in review_views:
        preview = ctx.deps.store.new_file(ctx.deps.chat_id, f"{tag}_{camera}.png")
        from mesh_forge.render import render_mesh_preview

        render_mesh_preview(
            src,
            preview,
            camera=camera,
            zoom=max(float(zoom or 1.0), 1.25 if camera == focus_view else 1.0),
            mesh=mesh,
            face_mask=mask,
        )
        images.append((f"{tag} / {camera}", preview))
        if camera == focus_view:
            art = ctx.deps.store.artifact_from_path(
                ctx.deps.chat_id,
                preview,
                label=f"candidate {tag}",
                view=camera,
            )
            ctx.deps.emit_artifact(art)
    verdict = LMStudioClient().review_mesh_mask(images, target=target) or {}
    kind = str(verdict.get("verdict") or "")
    score = _candidate_review_score(kind, float(verdict.get("confidence") or 0.0))
    return kind, score


def _candidate_review_score(verdict: str, confidence: float) -> float:
    rank = {
        "ok": 4.0,
        "too_little": 2.5,
        "partial": 2.2,
        "too_much": 1.0,
        "wrong": 0.5,
        "tiny_spot": 0.4,
        "missed": 0.2,
        "": 0.0,
    }
    return rank.get(str(verdict or "").strip().lower(), 0.0) + max(0.0, min(float(confidence or 0.0), 1.0))


def _remove_log(ctx: RunContext[ChatDeps], text: str) -> None:
    line = (text or "").strip()
    if line:
        ctx.deps.emit_event("tool_text_delta", delta=f"{line}\n")


def _emit_remove_debug_report(ctx: RunContext[ChatDeps], payload: dict[str, object]) -> None:
    dest = ctx.deps.store.new_file(ctx.deps.chat_id, "remove_extra_debug.json")
    dest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    art = ctx.deps.store.artifact_from_path(ctx.deps.chat_id, dest, label="remove extra debug")
    ctx.deps.emit_artifact(art)
