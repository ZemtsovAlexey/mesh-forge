from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel
from pydantic_ai import RunContext

from mesh_forge.adapters import ComfyUiClient
from mesh_forge.agent.deps import ChatDeps
from mesh_forge.domain import ImageArtifact, ImageSet
from mesh_forge.tools.base import MeshTool
from mesh_forge.tools.knobs import MeshGenKnobs, ViewName, apply_mesh_knobs


class ImageRef(BaseModel):
    ref: str
    view: ViewName | None = None


class ImagesToMesh(MeshTool):
    title = "Сборка mesh"
    heavy = True

    def run(
        self,
        ctx: RunContext[ChatDeps],
        images: list[ImageRef] | None = None,
        seed: int | None = None,
        quality: str | None = None,
        steps: int | None = None,
        cfg: float | None = None,
        guidance: float | None = None,
    ) -> str:
        """Reconstruct STL from N photos (1, 2, 3, or 4). Hunyuan MV max is 4 named cameras.

        Pass explicit refs. If images is empty, use this-turn photo attachments only (not chat history).
        1 image → single-view Hunyuan. 2–4 → multiview with ONLY provided slots (no front-padding).
        >4: first 4 used, rest reported as dropped. Label views after look if you know front/left/back/right.

        Bad mesh → quality=quality or higher steps, then retry. Redo → new seed.
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
        mesh = client.run_images_to_mesh(
            ImageSet(items=items),
            work,
            project_id=ctx.deps.chat_id,
            seed=echo["seed"],
        )
        dest = ctx.deps.store.new_file(ctx.deps.chat_id, "mesh.stl")
        dest.write_bytes(mesh.path.read_bytes())
        ctx.deps.store.set_current_mesh(ctx.deps.chat_id, dest)
        art = ctx.deps.store.artifact_from_path(ctx.deps.chat_id, dest, label="mesh")
        ctx.deps.emit_artifact(art)
        drop_note = f" Dropped extra: {dropped}." if dropped else ""
        views = ", ".join(f"{label}={path.name}" for label, path in picked)
        return f"Mesh {art.name} from {len(picked)} image(s): {views}.{drop_note} knobs={echo}"


def _pick_images(
    ctx: RunContext[ChatDeps],
    images: list[ImageRef] | None,
) -> tuple[list[tuple[str, Path]], list[str]]:
    labels = ("front", "left", "back", "right")
    selected: list[tuple[str, Path]] = []
    dropped: list[str] = []
    if images:
        for item in images:
            try:
                path = ctx.deps.store.resolve_ref(ctx.deps.chat_id, item.ref)
            except FileNotFoundError:
                dropped.append(item.ref)
                continue
            selected.append((item.view or "", path))
    else:
        photos = [a for a in ctx.deps.attachments if a.kind == "image"]
        for art in photos:
            selected.append(("", ctx.deps.store.resolve_file(ctx.deps.chat_id, art.name)))
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
