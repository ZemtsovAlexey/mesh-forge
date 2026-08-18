from __future__ import annotations

from pydantic_ai import RunContext

from mesh_forge.adapters import ComfyUiClient
from mesh_forge.agent.deps import ChatDeps
from mesh_forge.mesh_qc import mesh_is_usable
from mesh_forge.ops.geometry import load_mesh, save_mesh
from mesh_forge.render import render_mesh_front_clay
from mesh_forge.tools.base import MeshTool
from mesh_forge.tools.common import LOOK_AFTER, emit_mesh_preview, resolve_mesh, save_image_artifact
from mesh_forge.tools.knobs import ImageKnobs, apply_image_knobs


class RegenMesh(MeshTool):
    title = "Перегенерация"
    heavy = True

    def run(
        self,
        ctx: RunContext[ChatDeps],
        prompt: str,
        seed: int | None = None,
        quality: str | None = None,
        mesh_ref: str | None = None,
    ) -> str:
        """Rebuild the SAME object from the current mesh (not a new character).

        Bakes a clay front from the current STL, mild img2img, cuts the studio, Hunyuan from that ONE front
        (no Zero123 orbits — those invent broken sides). prompt: English, SAME object after the change.
        prompt: English description of this object AFTER the change (keep identity, pose, style).
        Do not call generate_image / images_to_mesh — that starts from scratch.
        If the result is a different object: restore_mesh(to='previous') and retry with a milder prompt.
        """
        from mesh_forge import progress as prog

        brief = (prompt or "").strip()
        if not brief:
            return "regen_mesh needs prompt: English, same object after the change."
        src = resolve_mesh(ctx, mesh_ref)
        knobs = ImageKnobs(
            seed=seed,
            quality=quality if quality in {"draft", "quality"} else None,
            style="clay",
        )
        cfg_obj, echo = apply_image_knobs(knobs)
        client = ComfyUiClient()
        client.config = cfg_obj
        work = ctx.deps.files_dir() / "work" / "regen"
        work.mkdir(parents=True, exist_ok=True)
        anchor = work / "anchor_front.png"
        prog.start(ctx.deps.chat_id, "regen_mesh", "guided")
        try:
            render_mesh_front_clay(src, anchor, size=768)
        except Exception as exc:
            return f"regen_mesh failed to bake current mesh: {exc}"
        try:
            generated = client.run_guided_edit(
                brief,
                anchor,
                work,
                project_id=ctx.deps.chat_id,
                seed=echo["seed"],
            )
        except Exception as exc:
            return (
                f"regen_mesh failed: {exc}. Current STL is unchanged. "
                f"knobs={echo}"
            )
        for item in generated.views.items:
            save_image_artifact(
                ctx,
                item.path,
                label=item.label or "view",
                view=item.label or "",
            )
        dest = ctx.deps.store.new_file(ctx.deps.chat_id, "regen.stl")
        try:
            save_mesh(load_mesh(generated.mesh.path), dest)
        except Exception as exc:
            return (
                f"regen_mesh wrote a mesh but STL convert failed: {exc}. "
                "Current STL is unchanged."
            )
        ok, qc = mesh_is_usable(dest)
        if not ok:
            return (
                f"regen_mesh produced an empty/invalid mesh ({dest.name}). {qc} "
                "Current STL is unchanged. Retry with a new seed."
            )
        ctx.deps.store.set_current_mesh(ctx.deps.chat_id, dest, role="edit")
        art = ctx.deps.store.artifact_from_path(ctx.deps.chat_id, dest, label="regen")
        ctx.deps.emit_artifact(art)
        emit_mesh_preview(ctx, dest)
        return (
            f"Regenerated {src.name} → {art.name} keeping identity (clay bake + guided img2img). "
            f"knobs={echo} {LOOK_AFTER}"
        )
