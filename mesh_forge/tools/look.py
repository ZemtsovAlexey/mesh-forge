from __future__ import annotations

from pathlib import Path

from pydantic_ai import RunContext

from mesh_forge.adapters import LMStudioClient
from mesh_forge.agent.deps import ChatDeps
from mesh_forge.render import render_mesh_preview
from mesh_forge.tools.base import MeshTool
from mesh_forge.tools.common import resolve_mesh


class Look(MeshTool):
    title = "Смотрю"
    heavy = True

    def run(
        self,
        ctx: RunContext[ChatDeps],
        target: str = "auto",
        question: str = "",
        refs: list[str] | None = None,
    ) -> str:
        """Vision look at images or a mesh preview. target: auto|mesh|images. refs: artifact ids; omit to use this-turn photos or latest images/mesh."""
        store = ctx.deps.store
        chat_id = ctx.deps.chat_id
        images: list[tuple[str, Path]] = []
        wanted = (target or "auto").strip().lower()
        if refs:
            for ref in refs[:4]:
                try:
                    path = store.resolve_ref(chat_id, ref)
                except FileNotFoundError:
                    continue
                images.append((ref, path))
        elif wanted in {"auto", "images"}:
            attached = [a for a in ctx.deps.attachments if a.kind == "image"]
            for art in attached[:4]:
                images.append((art.label or art.id, store.resolve_file(chat_id, art.name)))
            if not images:
                pics = [a for a in store.list_files(chat_id) if a.kind == "image"][-4:]
                for art in pics:
                    images.append((art.label or art.id, store.resolve_file(chat_id, art.name)))
        if wanted == "mesh" or (wanted == "auto" and not images):
            mesh_path = resolve_mesh(ctx)
            preview = store.new_file(chat_id, "look_mesh.png")
            render_mesh_preview(mesh_path, preview)
            art = store.artifact_from_path(chat_id, preview, label="mesh preview")
            ctx.deps.emit_artifact(art)
            images = [("mesh", preview)]
        if not images:
            return "Нечего смотреть: нет картинок и нет mesh."
        vision_inputs: list[tuple[str, Path]] = []
        for label, path in images:
            if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
                vision_inputs.append((label, path))
            elif path.suffix.lower() in {".stl", ".obj"}:
                preview = store.new_file(chat_id, f"look_{path.stem}.png")
                render_mesh_preview(path, preview)
                vision_inputs.append((label, preview))
        if not vision_inputs:
            return "Нечего смотреть."
        note = LMStudioClient().inspect_images(vision_inputs[:4], question=question)
        return note or "Vision model returned empty description."
