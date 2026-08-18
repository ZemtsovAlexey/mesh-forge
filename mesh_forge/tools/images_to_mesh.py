from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator, model_validator
from pydantic_ai import RunContext

from mesh_forge.adapters import ComfyUiClient
from mesh_forge.agent.deps import ChatDeps
from mesh_forge.domain import ImageArtifact, ImageSet
from mesh_forge.mesh_qc import mesh_is_usable
from mesh_forge.ops.geometry import load_mesh, save_mesh
from mesh_forge.tools.base import MeshTool
from mesh_forge.tools.common import emit_mesh_preview
from mesh_forge.tools.knobs import MeshGenKnobs, ViewName, apply_mesh_knobs

_VIEW_NAMES = ("front", "left", "back", "right")


def _view_from_name(value: str) -> str | None:
    stem = Path(str(value or "")).stem.strip().lower()
    for view in _VIEW_NAMES:
        if stem == view or stem.endswith(f"_{view}") or stem.endswith(f"-{view}"):
            return view
    return None


class ImageRef(BaseModel):
    ref: str
    view: ViewName | None = None

    @model_validator(mode="before")
    @classmethod
    def coerce_ref(cls, data: Any) -> Any:
        if isinstance(data, str):
            return {"ref": data, "view": _view_from_name(data)}
        if isinstance(data, dict):
            payload = dict(data)
            if not payload.get("ref"):
                alt = payload.get("id") or payload.get("name") or payload.get("image")
                if alt:
                    payload["ref"] = alt
            if not payload.get("view"):
                payload["view"] = _view_from_name(str(payload.get("ref") or ""))
            return payload
        return data


def _coerce_image_list(value: Any) -> Any:
    if value is None or value == "":
        return None
    if isinstance(value, (str, dict)):
        return [value]
    return value


class ImagesToMesh(MeshTool):
    title = "Сборка mesh"
    heavy = True

    def run(
        self,
        ctx: RunContext[ChatDeps],
        images: Annotated[list[ImageRef] | None, BeforeValidator(_coerce_image_list)] = None,
        seed: int | None = None,
        quality: str | None = None,
        steps: int | None = None,
        cfg: float | None = None,
        guidance: float | None = None,
    ) -> str:
        """Reconstruct STL from N photos (1, 2, 3, or 4). Hunyuan MV max is 4 named cameras.

        images: artifact ids as strings, e.g. ["a19885e6_front"]. Objects {ref, view} also work.
        If images is empty, use this-turn photo attachments, else images from a chat reply.
        1 image → single-view Hunyuan. 2–4 → multiview with ONLY provided slots (no front-padding).
        Photo with floor/studio/background → call remove_background first and pass the cutout ids, not the originals.
        Several generate_image fronts are NOT left/back/right — use one photo, or generate_views for a real orbit.
        >4: first 4 used, rest reported as dropped. Label views after look if you know front/left/back/right.

        Empty mesh (0 verts) is a failed reconstruction, not a hole. Retry with one front photo and a new seed.
        Open/non-watertight is normal for Hunyuan. To undo a bad geometry edit use restore_mesh (previous or source).
        """
        from mesh_forge import progress as prog

        picked, dropped = _pick_images(ctx, images)
        if not picked:
            return "No images. Pass refs or attach photos on this turn."
        knobs = MeshGenKnobs(
            seed=seed,
            quality=quality if quality in {"draft", "quality"} else None,
            steps=steps,
            cfg=cfg,
            guidance=guidance,
        )
        cfg_obj, echo = apply_mesh_knobs(knobs)
        client = ComfyUiClient()
        client.config = cfg_obj
        items = [
            ImageArtifact(path=path, label=label, role="view", stage="views")
            for label, path in picked
        ]
        work = ctx.deps.files_dir() / "work"
        prog.start(ctx.deps.chat_id, "images_to_mesh", "mesh")
        try:
            mesh = client.run_images_to_mesh(
                ImageSet(items=items),
                work,
                project_id=ctx.deps.chat_id,
                seed=echo["seed"],
            )
        except Exception as exc:
            views = ", ".join(f"{label}={path.name}" for label, path in picked)
            return (
                f"ERROR: Hunyuan failed on {len(picked)} image(s): {views}. {exc}. "
                "If a previous mesh exists, restore_mesh(to='source'). "
                "Else retry images_to_mesh with ONE front photo and a new seed."
                f" knobs={echo}"
            )
        dest = ctx.deps.store.new_file(ctx.deps.chat_id, "mesh.stl")
        try:
            save_mesh(load_mesh(mesh.path), dest)
        except Exception as exc:
            return (
                f"ERROR: Hunyuan wrote {mesh.path.name}, but it could not be converted to STL: {exc}. "
                "If a previous mesh exists, restore_mesh(to='source'). "
                "Else retry images_to_mesh with ONE front photo and a new seed."
            )
        ok, qc = mesh_is_usable(dest)
        views = ", ".join(f"{label}={path.name}" for label, path in picked)
        drop_note = f" Dropped extra: {dropped}." if dropped else ""
        if not ok:
            return (
                f"ERROR: Hunyuan produced an empty/invalid mesh ({dest.name}) from {len(picked)} image(s): {views}."
                f"{drop_note} {qc} If a previous mesh exists, restore_mesh(to='source'). "
                f"Else retry images_to_mesh with ONE front photo "
                f"(not several generate_image fronts as left/back/right) and a new seed. knobs={echo}"
            )
        ctx.deps.store.set_current_mesh(ctx.deps.chat_id, dest, role="source")
        art = ctx.deps.store.artifact_from_path(ctx.deps.chat_id, dest, label="mesh")
        ctx.deps.emit_artifact(art)
        emit_mesh_preview(ctx, dest)
        return (
            f"Mesh {art.name} from {len(picked)} image(s): {views}.{drop_note} knobs={echo} "
            "Это source-меш. Остановись. Не inspect+repair+smooth, пока пользователь не попросит. "
            "Открытая поверхность — норма. Если позже правка испортит форму — restore_mesh."
        )


def _pick_images(
    ctx: RunContext[ChatDeps],
    images: list[ImageRef] | None,
) -> tuple[list[tuple[str, Path]], list[str]]:
    labels = ("front", "left", "back", "right")
    selected: list[tuple[str, Path]] = []
    dropped: list[str] = []
    wanted = list(images or [])
    if wanted:
        views = [(item.view or "").strip().lower() for item in wanted]
        labeled = [v for v in views if v]
        if len(wanted) > 1 and labeled and len(set(labeled)) == 1:
            dropped.extend(item.ref for item in wanted[:-1])
            wanted = wanted[-1:]
        for item in wanted:
            try:
                path = ctx.deps.store.resolve_ref(ctx.deps.chat_id, item.ref)
            except FileNotFoundError:
                dropped.append(item.ref)
                continue
            selected.append((item.view or _view_from_name(item.ref) or _view_from_name(path.name) or "", path))
    else:
        photos = [a for a in ctx.deps.attachments if a.kind == "image"]
        if not photos:
            photos = [a for a in ctx.deps.reply_artifacts if a.kind == "image"]
        for art in photos:
            selected.append(
                (
                    (art.view or art.label or "").strip().lower(),
                    ctx.deps.store.resolve_file(ctx.deps.chat_id, art.name),
                )
            )
    if len(selected) > 1 and all("_front" in path.name.lower() for _, path in selected):
        dropped.extend(path.name for _, path in selected[:-1])
        last_label, last_path = selected[-1]
        selected = [(last_label or "front", last_path)]
    if len(selected) > 4:
        dropped.extend(name for _, p in selected[4:] for name in [p.name])
        selected = selected[:4]
    named: list[tuple[str, Path]] = []
    used = set()
    unlabeled: list[Path] = []
    for label, path in selected:
        key = (label or "").strip().lower()
        if key in labels and key not in used:
            named.append((key, path))
            used.add(key)
        else:
            unlabeled.append(path)
    for path in unlabeled:
        for label in labels:
            if label not in used:
                named.append((label, path))
                used.add(label)
                break
    return named, dropped
