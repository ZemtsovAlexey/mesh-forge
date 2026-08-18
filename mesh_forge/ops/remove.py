from __future__ import annotations

from typing import Any

import numpy as np
import trimesh

from mesh_forge.ops.edit import EditError, _slice_keep_plane
from mesh_forge.ops.geometry import carve_faces, carve_region, resolve_carve_box
from mesh_forge.ops.region import faces_in_box, knife_lump_faces, knife_plane
from mesh_forge.ops.topo import mask_aim_side


def classify_removal_strategy(describe: str) -> str:
    text = (describe or "").strip().lower()
    if any(word in text for word in ("юбк", "подол", "hem", "skirt")) and any(
        word in text for word in ("лепест", "отрост", "petal", "flap")
    ):
        return "hem_flap_trim"
    if any(word in text for word in ("полукруг", "подол", "край", "кромк", "trim", "hem", "edge")):
        return "edge_trim"
    if any(word in text for word in ("остров", "island", "detached", "отдельн", "float", "мусор")):
        return "island_drop"
    if any(word in text for word in ("пятн", "patch", "поверхност", "заплат", "spot", "панел")):
        return "surface_patch"
    return "protrusion_cut"


def build_auto_remove_proposal(mesh: trimesh.Trimesh, describe: str) -> dict[str, Any]:
    strategy = classify_removal_strategy(describe)
    if strategy == "hem_flap_trim":
        return _build_hem_flap_trim(mesh, describe)
    if strategy == "protrusion_cut":
        return _build_protrusion_cut(mesh, describe)
    if strategy == "island_drop":
        return _build_island_drop(mesh, describe)
    if strategy == "edge_trim":
        return _build_edge_trim(mesh, describe)
    return {"strategy": "surface_patch", "need_mask": True, "note": "Need mask-based surface patch strategy."}


def _build_protrusion_cut(mesh: trimesh.Trimesh, describe: str) -> dict[str, Any]:
    side = _strategy_side(describe)
    along = _strategy_along(describe)
    masks = _protrusion_candidates(mesh, side, along=along, describe=describe)
    ranked = _ranked_protrusion_candidates(mesh, masks, side=side, describe=describe)
    mask = np.asarray(ranked[0], dtype=bool) if ranked else np.zeros(int(len(mesh.faces)), dtype=bool)
    if not np.any(mask):
        raise EditError("Automatic protrusion cut found no removable lump.")
    out, stats = carve_faces(mesh, mask, min_keep_ratio=0.50, min_keep_faces=8, drop_crumbs=False)
    return {
        "strategy": "protrusion_cut",
        "mask": np.asarray(mask, dtype=bool),
        "candidate_masks": [np.asarray(item, dtype=bool) for item in ranked[:3]],
        "mesh": out,
        "note": f"Auto strategy: protrusion cut on {side}" + (f" / {along}" if along else "") + ".",
        "stats": stats,
    }


def _build_island_drop(mesh: trimesh.Trimesh, describe: str) -> dict[str, Any]:
    comps = _face_components(mesh)
    n_faces = int(len(mesh.faces))
    if len(comps) < 2:
        raise EditError("No disconnected extra island was found.")
    comps = [c for c in comps if 0 < len(c) < n_faces]
    if not comps:
        raise EditError("No disconnected removable island was found.")
    comps.sort(key=len)
    idx = comps[0]
    if len(idx) > max(24, int(0.25 * max(n_faces, 1))):
        raise EditError("The smallest disconnected part is too large to auto-drop safely.")
    mask = np.zeros(n_faces, dtype=bool)
    mask[np.asarray(idx, dtype=np.int64)] = True
    out, stats = carve_faces(mesh, mask, min_keep_ratio=0.50, min_keep_faces=8, drop_crumbs=False)
    return {
        "strategy": "island_drop",
        "mask": mask,
        "mesh": out,
        "note": "Auto strategy: drop disconnected island.",
        "stats": stats,
    }


def _build_edge_trim(mesh: trimesh.Trimesh, describe: str) -> dict[str, Any]:
    side = _strategy_side(describe)
    box = _edge_trim_box(describe, side)
    mask = faces_in_box(mesh, box)
    if np.any(mask):
        out, stats = carve_region(
            mesh,
            box,
            action="remove",
            min_keep_ratio=0.50,
            min_keep_faces=24,
            drop_crumbs=False,
            protect_sides=False,
        )
        return {
            "strategy": "edge_trim",
            "mask": np.asarray(mask, dtype=bool),
            "mesh": out,
            "note": f"Auto strategy: edge trim from {side}.",
            "stats": stats,
        }
    origin, normal, label = knife_plane(mesh, side, at=_edge_plane_at(side))
    out = _slice_keep_plane(mesh, origin, normal)
    if len(out.faces) <= 0:
        raise EditError("Edge trim found no removable segment.")
    return {
        "strategy": "edge_trim",
        "mask": np.zeros(int(len(mesh.faces)), dtype=bool),
        "mesh": out,
        "note": f"Auto strategy: plane trim {label}.",
        "stats": {"region": label, "faces_before": int(len(mesh.faces)), "faces_after": int(len(out.faces))},
    }


def _build_hem_flap_trim(mesh: trimesh.Trimesh, describe: str) -> dict[str, Any]:
    side = _strategy_side(describe)
    candidates: list[np.ndarray] = []
    seen: set[bytes] = set()
    for box in _hem_flap_boxes(describe, side):
        mask = np.asarray(faces_in_box(mesh, box), dtype=bool)
        key = mask.tobytes()
        if np.any(mask) and key not in seen:
            candidates.append(mask)
            seen.add(key)
    ranked0 = _ranked_protrusion_candidates(mesh, candidates, side=side, describe=describe)
    if len(ranked0) >= 2:
        union = np.zeros_like(ranked0[0], dtype=bool)
        for index, mask in enumerate(ranked0[:4], start=1):
            union |= np.asarray(mask, dtype=bool)
            key = union.tobytes()
            if index >= 2 and np.any(union) and key not in seen:
                candidates.append(np.asarray(union, dtype=bool))
                seen.add(key)
    ranked = _ranked_protrusion_candidates(mesh, candidates, side=side, describe=describe)
    mask = np.asarray(ranked[0], dtype=bool) if ranked else np.zeros(int(len(mesh.faces)), dtype=bool)
    if not np.any(mask):
        raise EditError("Automatic hem flap trim found no removable hem segment.")
    out, stats = carve_faces(mesh, mask, min_keep_ratio=0.55, min_keep_faces=24, drop_crumbs=False)
    return {
        "strategy": "hem_flap_trim",
        "mask": mask,
        "candidate_masks": [np.asarray(item, dtype=bool) for item in ranked[:3]],
        "mesh": out,
        "note": f"Auto strategy: hem flap trim from {side}.",
        "stats": stats,
    }


def _strategy_side(describe: str) -> str:
    text = (describe or "").lower()
    if any(word in text for word in ("сзади", "зад", "back")):
        return "back"
    if any(word in text for word in ("спереди", "перед", "front")):
        return "front"
    if any(word in text for word in ("слева", "лев", "left")):
        return "left"
    if any(word in text for word in ("сверху", "верх", "top")):
        return "top"
    if any(word in text for word in ("снизу", "низ", "bottom")):
        return "bottom"
    side = mask_aim_side(describe, None)
    return side if side in {"left", "right"} else "right"


def _strategy_along(describe: str) -> str:
    text = (describe or "").lower()
    if any(word in text for word in ("подол", "юбк", "низ", "leg", "ножк", "petal", "лепест")):
        return "bottom"
    if any(word in text for word in ("верх", "голов", "top")):
        return "top"
    if any(word in text for word in ("спереди", "перед", "front")):
        return "front"
    if any(word in text for word in ("сзади", "зад", "back")):
        return "back"
    return ""


def _protrusion_candidates(
    mesh: trimesh.Trimesh,
    side: str,
    *,
    along: str,
    describe: str,
) -> list[np.ndarray]:
    candidates: list[np.ndarray] = []
    seen: set[bytes] = set()
    for name in _candidate_alongs(along, describe):
        mask = knife_lump_faces(mesh, side, along=name, hops=18)
        key = np.asarray(mask, dtype=bool).tobytes()
        if np.any(mask) and key not in seen:
            candidates.append(np.asarray(mask, dtype=bool))
            seen.add(key)
    if _skirt_like(describe):
        seed_masks: list[np.ndarray] = []
        for seed in _semantic_tip_seeds(mesh, side, describe, limit=6):
            mask = _grow_lump_from_tip(mesh, side, seed)
            key = np.asarray(mask, dtype=bool).tobytes()
            if np.any(mask) and key not in seen:
                current = np.asarray(mask, dtype=bool)
                candidates.append(current)
                seed_masks.append(current)
                seen.add(key)
        if len(seed_masks) >= 2:
            union = np.zeros_like(seed_masks[0], dtype=bool)
            for index, mask in enumerate(seed_masks[:4], start=1):
                union |= np.asarray(mask, dtype=bool)
                key = union.tobytes()
                if index >= 2 and np.any(union) and key not in seen:
                    candidates.append(np.asarray(union, dtype=bool))
                    seen.add(key)
    return candidates


def _candidate_alongs(along: str, describe: str) -> list[str]:
    order: list[str] = []
    if along:
        order.append(along)
    if _skirt_like(describe):
        order.extend(["bottom", "front", "back", ""])
    else:
        order.append("")
    out: list[str] = []
    for item in order:
        if item not in out:
            out.append(item)
    return out


def _best_protrusion_candidate(
    mesh: trimesh.Trimesh,
    candidates: list[np.ndarray],
    *,
    side: str,
    describe: str,
) -> np.ndarray:
    ranked = _ranked_protrusion_candidates(mesh, candidates, side=side, describe=describe)
    if not ranked:
        return np.zeros(int(len(mesh.faces)), dtype=bool)
    return np.asarray(ranked[0], dtype=bool)


def _ranked_protrusion_candidates(
    mesh: trimesh.Trimesh,
    candidates: list[np.ndarray],
    *,
    side: str,
    describe: str,
) -> list[np.ndarray]:
    if not candidates:
        return []
    ranked = sorted(
        (np.asarray(mask, dtype=bool) for mask in candidates),
        key=lambda mask: _semantic_protrusion_score(mesh, mask, side=side, describe=describe),
        reverse=True,
    )
    return ranked


def _semantic_protrusion_score(
    mesh: trimesh.Trimesh,
    mask: np.ndarray,
    *,
    side: str,
    describe: str,
) -> float:
    current = np.asarray(mask, dtype=bool)
    if not np.any(current):
        return -1e18
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    lo, hi = bounds[0], bounds[1]
    span = np.maximum(hi - lo, 1e-9)
    centers = np.asarray(mesh.triangles_center, dtype=np.float64)
    norm = (centers - lo) / span
    pts = norm[current]
    side_axis, side_sign = _side_axis_sign(side)
    side_mean = float(np.mean(pts[:, side_axis] * float(side_sign)))
    size_frac = float(np.mean(current))
    score = 2.0 * side_mean - 0.60 * size_frac
    if _skirt_like(describe):
        y_lo = float(np.percentile(pts[:, 1], 20))
        y_mean = float(np.mean(pts[:, 1]))
        score += 2.20 * (1.0 - y_lo) + 1.20 * (1.0 - y_mean)
    if "front" in _strategy_along(describe):
        score += 0.50 * float(np.mean(pts[:, 2]))
    if "back" in _strategy_along(describe):
        score += 0.50 * float(np.mean(1.0 - pts[:, 2]))
    return score


def _semantic_tip_seeds(
    mesh: trimesh.Trimesh,
    side: str,
    describe: str,
    *,
    limit: int,
) -> list[int]:
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    if len(verts) <= 0:
        return []
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    lo, hi = bounds[0], bounds[1]
    span = np.maximum(hi - lo, 1e-9)
    norm = (verts - lo) / span
    axis, sign = _side_axis_sign(side)
    mask = np.ones(len(verts), dtype=bool)
    if sign > 0:
        mask &= norm[:, axis] >= 0.52
    else:
        mask &= norm[:, axis] <= 0.48
    if _skirt_like(describe):
        mask &= norm[:, 1] <= 0.55
    if not np.any(mask):
        mask = np.ones(len(verts), dtype=bool)
    score = norm[:, axis] * float(sign)
    if _skirt_like(describe):
        score += 0.75 * (1.0 - norm[:, 1])
    if any(word in (describe or "").lower() for word in ("спереди", "перед", "front")):
        score += 0.20 * norm[:, 2]
    if any(word in (describe or "").lower() for word in ("сзади", "зад", "back")):
        score += 0.20 * (1.0 - norm[:, 2])
    idx = np.where(mask)[0]
    ranked = idx[np.argsort(score[idx])[::-1]]
    return [int(i) for i in ranked[: max(1, int(limit))]]


def _grow_lump_from_tip(mesh: trimesh.Trimesh, side: str, tip: int, *, hops: int = 18) -> np.ndarray:
    n = int(len(mesh.faces))
    out = np.zeros(n, dtype=bool)
    if n <= 0 or int(len(mesh.vertices)) <= 0 or tip < 0 or tip >= int(len(mesh.vertices)):
        return out
    axis, sign = _side_axis_sign(side)
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    scores = verts[:, axis] * float(sign)
    tip_score = float(scores[int(tip)])
    ordered = np.sort(scores)
    spread = float(ordered[-1] - ordered[0]) or 1.0
    k = max(1, int(0.45 * len(ordered)))
    shell = float(np.percentile(scores, 88.0))
    if k < len(ordered) - 2:
        window = ordered[k:]
        gaps = np.diff(window)
        j = int(np.argmax(gaps))
        if float(gaps[j]) > 0.06 * spread:
            shell = float(window[j])
    inward_lim = float(np.percentile(scores, 80.0)) if tip_score <= shell + 1e-12 else shell + 0.04 * (
        tip_score - shell
    )
    nbrs: list[set[int]] = [set() for _ in range(len(verts))]
    for a, b, c in faces:
        ia, ib, ic = int(a), int(b), int(c)
        nbrs[ia].update((ib, ic))
        nbrs[ib].update((ia, ic))
        nbrs[ic].update((ia, ib))
    lump = np.zeros(len(verts), dtype=bool)
    lump[int(tip)] = True
    current = [int(tip)]
    limit = max(4, min(int(hops), 28))
    for _ in range(limit):
        nxt: list[int] = []
        for v in current:
            for n_i in nbrs[v]:
                if lump[n_i]:
                    continue
                if float(scores[n_i]) < inward_lim:
                    continue
                nxt.append(n_i)
        if not nxt:
            break
        uniq = list(dict.fromkeys(nxt))
        if int(lump.sum()) >= 12:
            ring_score = float(np.median(scores[np.asarray(uniq, dtype=np.int64)]))
            entered_body = ring_score <= shell + 0.04 * max(tip_score - shell, 1e-9)
            exploded = len(uniq) >= max(24, int(3.5 * len(current)))
            if entered_body and exploded:
                break
        for n_i in uniq:
            lump[n_i] = True
        current = uniq
    counts = lump[faces].sum(axis=1)
    out = counts >= 2
    if not np.any(out):
        out = np.any(lump[faces], axis=1)
    return out


def _side_axis_sign(side: str) -> tuple[int, int]:
    return {
        "right": (0, 1),
        "left": (0, -1),
        "top": (1, 1),
        "bottom": (1, -1),
        "front": (2, 1),
        "back": (2, -1),
    }.get(str(side or "").strip().lower(), (0, 1))


def _skirt_like(describe: str) -> bool:
    text = (describe or "").lower()
    return any(word in text for word in ("юбк", "подол", "hem", "skirt", "лепест"))


def _edge_trim_box(describe: str, side: str) -> tuple[float, float, float, float, float, float]:
    text = (describe or "").lower()
    amount = 0.14 if any(word in text for word in ("маленьк", "small", "кусоч")) else 0.18
    kwargs: dict[str, float | str] = {"side": side, "amount": amount}
    if any(word in text for word in ("подол", "юбк", "низ", "hem")):
        kwargs["bottom"] = 0.0
        kwargs["top"] = 0.34
    if any(word in text for word in ("сзади", "зад", "back")):
        kwargs["back"] = 0.0
        kwargs["front"] = 0.52
    elif any(word in text for word in ("спереди", "перед", "front")):
        kwargs["back"] = 0.48
        kwargs["front"] = 1.0
    else:
        kwargs["back"] = 0.32
        kwargs["front"] = 0.92
    return resolve_carve_box(**kwargs)


def _edge_plane_at(side: str) -> float:
    if side in {"left", "back", "bottom"}:
        return 0.18
    return 0.82


def _hem_flap_boxes(describe: str, side: str) -> list[tuple[float, float, float, float, float, float]]:
    text = (describe or "").lower()
    base_amount = 0.14 if any(word in text for word in ("маленьк", "small", "кусоч")) else 0.18
    amounts = []
    for value in (max(0.10, base_amount - 0.04), base_amount, min(0.24, base_amount + 0.04)):
        if value not in amounts:
            amounts.append(value)
    boxes: list[tuple[float, float, float, float, float, float]] = []
    if side in {"right", "left"}:
        z_windows = [(0.00, 0.38), (0.14, 0.56), (0.32, 0.74), (0.52, 1.00), (0.20, 0.92)]
        if any(word in text for word in ("спереди", "перед", "front")):
            z_windows = [(0.52, 1.00), (0.32, 0.74), (0.20, 0.92)]
        elif any(word in text for word in ("сзади", "зад", "back")):
            z_windows = [(0.00, 0.38), (0.14, 0.56), (0.20, 0.92)]
        for amount in amounts:
            for back, front in z_windows:
                boxes.append(
                    resolve_carve_box(
                        side=side,
                        amount=amount,
                        bottom=0.00,
                        top=0.42,
                        back=back,
                        front=front,
                    )
                )
        return boxes
    x_windows = [(0.00, 0.38), (0.14, 0.56), (0.32, 0.74), (0.52, 1.00), (0.20, 0.92)]
    if any(word in text for word in ("слева", "лев", "left")):
        x_windows = [(0.00, 0.38), (0.14, 0.56), (0.20, 0.92)]
    elif any(word in text for word in ("справа", "прав", "right")):
        x_windows = [(0.52, 1.00), (0.32, 0.74), (0.20, 0.92)]
    for amount in amounts:
        for left, right in x_windows:
            boxes.append(
                resolve_carve_box(
                    side=side,
                    amount=amount,
                    left=left,
                    right=right,
                    bottom=0.00,
                    top=0.42,
                )
            )
    return boxes


def _face_components(mesh: trimesh.Trimesh) -> list[np.ndarray]:
    n = int(len(mesh.faces))
    if n <= 0:
        return []
    raw = getattr(mesh, "face_adjacency", None)
    if raw is None or len(raw) == 0:
        return [np.arange(n, dtype=np.int64)]
    try:
        comps = list(
            trimesh.graph.connected_components(
                np.asarray(raw, dtype=np.int64),
                min_len=1,
                nodes=np.arange(n, dtype=np.int64),
            )
        )
    except Exception:
        return [np.arange(n, dtype=np.int64)]
    return [np.asarray(comp, dtype=np.int64) for comp in comps]
