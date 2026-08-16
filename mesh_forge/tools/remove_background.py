from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

from pydantic import BeforeValidator
from pydantic_ai import RunContext

from mesh_forge.agent.deps import ChatDeps
from mesh_forge.ops.background import cut_background, has_alpha
from mesh_forge.tools.base import MeshTool
from mesh_forge.tools.common import save_image_artifact

_VIEW_NAMES = ("front", "left", "back", "right")


def _coerce_image_list(value: Any) -> Any:
    if value is None or value == "":
        return None
    if isinstance(value, str):
        return [value]
    return value


def _view_from_name(value: str) -> str | None:
    stem = Path(str(value or "")).stem.strip().lower()
    for view in _VIEW_NAMES:
        if stem == view or stem.endswith(f"_{view}") or stem.endswith(f"-{view}"):
            return view
    return None


class RemoveBackground(MeshTool):
    title = "Фон"
    heavy = True

    def run(
        self,
        ctx: RunContext[ChatDeps],
        images: Annotated[list[str] | None, BeforeValidator(_coerce_image_list)] = None,
    ) -> str:
        """Cut the subject out of photos (transparent PNG). Use before images_to_mesh when the shot has floor/studio/background.

        images: artifact ids as strings, e.g. ["a19885e6_front"]. Empty = this-turn attachments, else reply images.
        Clay from generate_image: skip this tool, go straight to images_to_mesh.
        After cutout: look at the PNG (thin legs/holes still there?), then images_to_mesh with the NEW ids, not the originals.
        Already-transparent images are skipped. Do not repair a mesh cube/panel leftover — recut and remesh.
        """
        from mesh_forge import progress as prog

        picked, dropped = _pick_images(ctx, images)
        if not picked:
            return "No images. Pass refs or attach photos on this turn."
        work = ctx.deps.files_dir() / "work"
        work.mkdir(parents=True, exist_ok=True)
        prog.start(ctx.deps.chat_id, "remove_background", "cutout")
        made: list[str] = []
        skipped: list[str] = []
        errors: list[str] = []
        total = len(picked)
        try:
            for idx, (label, path) in enumerate(picked, start=1):
                prog.update(
                    ctx.deps.chat_id,
                    10 + 80 * idx / max(total, 1),
                    "cutout",
                )
                view = label or _view_from_name(path.name) or ""
                if has_alpha(path):
                    skipped.append(path.name)
                    continue
                tmp = work / f"{path.stem}_cutout.png"
                try:
                    cut_background(path, tmp)
                except Exception as exc:
                    errors.append(f"{path.name}: {exc}")
                    continue
                art = save_image_artifact(
                    ctx,
                    tmp,
                    label=f"{view} cutout".strip() if view else "cutout",
                    view=view,
                )
                made.append(f"{art.name}" + (f" view={view}" if view else ""))
        finally:
            prog.finish(ctx.deps.chat_id, ok=not errors or bool(made))
        drop_note = f" Dropped extra: {dropped}." if dropped else ""
        skip_note = (
            f" Already transparent (use as-is): {', '.join(skipped)}." if skipped else ""
        )
        err_note = f" Failed: {'; '.join(errors)}." if errors else ""
        if not made and not skipped:
            return f"ERROR: rembg failed on {len(picked)} image(s).{err_note}{drop_note}"
        if not made:
            return (
                f"No new cutouts.{skip_note}{err_note}{drop_note} "
                "Pass those ids to images_to_mesh."
            )
        return (
            f"Cutout {len(made)}: {', '.join(made)}.{skip_note}{err_note}{drop_note} "
            "look at the PNG, then images_to_mesh with these new ids."
        )


def _pick_images(
    ctx: RunContext[ChatDeps],
    images: list[str] | None,
) -> tuple[list[tuple[str, Path]], list[str]]:
    selected: list[tuple[str, Path]] = []
    dropped: list[str] = []
    wanted = [str(item).strip() for item in (images or []) if str(item).strip()]
    if wanted:
        for ref in wanted:
            try:
                path = ctx.deps.store.resolve_ref(ctx.deps.chat_id, ref)
            except FileNotFoundError:
                dropped.append(ref)
                continue
            selected.append((_view_from_name(ref) or _view_from_name(path.name) or "", path))
    else:
        photos = [a for a in ctx.deps.attachments if a.kind == "image"]
        if not photos:
            photos = [a for a in ctx.deps.reply_artifacts if a.kind == "image"]
        for art in photos:
            selected.append(
                (
                    (art.view or "").strip().lower()
                    or _view_from_name(art.label or "")
                    or _view_from_name(art.name)
                    or "",
                    ctx.deps.store.resolve_file(ctx.deps.chat_id, art.name),
                )
            )
    if len(selected) > 4:
        dropped.extend(path.name for _, path in selected[4:])
        selected = selected[:4]
    return selected, dropped
