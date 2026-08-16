from __future__ import annotations

from pydantic_ai import RunContext

from mesh_forge.adapters import ComfyUiClient
from mesh_forge.agent.deps import ChatDeps
from mesh_forge.tools.base import MeshTool
from mesh_forge.tools.common import reply_image_id, save_image_artifact
from mesh_forge.tools.knobs import ImageKnobs, ViewName, apply_image_knobs


class GenerateViews(MeshTool):
    title = "Виды"
    heavy = True

    def run(
        self,
        ctx: RunContext[ChatDeps],
        prompt: str,
        views: list[ViewName] | None = None,
        ref_image: str | None = None,
        seed: int | None = None,
        quality: str | None = None,
        steps: int | None = None,
        cfg: float | None = None,
        style: str | None = None,
    ) -> str:
        """Generate named views. views defaults to front,left,back,right — pass a subset if you only need some.

        If ref_image is set (artifact id), Zero123 orbits from that front.
        If omitted, uses the image the user replied to (front if present). Else text→front, then Zero123.
        Front must be a true eye-level orthographic shot; a 3/4 or tilted front makes every orbit crooked.
        Redo a crooked front with generate_image first, then orbit. Redo orbits → new seed.
        """
        from mesh_forge import progress as prog

        wanted = views or ["front", "left", "back", "right"]
        ref_image = (ref_image or "").strip() or reply_image_id(ctx)
        knobs = ImageKnobs(
            seed=seed,
            quality=quality if quality in {"draft", "quality"} else None,
            steps=steps,
            cfg=cfg,
            style=style if style in {"clay", "color"} else None,
        )
        cfg_obj, echo = apply_image_knobs(knobs)
        client = ComfyUiClient()
        client.config = cfg_obj
        work = ctx.deps.files_dir() / "work"
        prog.start(ctx.deps.chat_id, "generate_views", "views")
        if ref_image:
            front = ctx.deps.store.resolve_ref(ctx.deps.chat_id, ref_image)
            result = client.generate_views_from_front(
                prompt,
                front,
                work,
                project_id=ctx.deps.chat_id,
                seed=echo["seed"],
            )
        elif set(wanted) == {"front"}:
            result = client.generate_front(
                prompt, work, project_id=ctx.deps.chat_id, seed=echo["seed"]
            )
        else:
            result = client.generate_views(
                prompt,
                work,
                count=max(4, len(wanted)),
                project_id=ctx.deps.chat_id,
                seed=echo["seed"],
            )
        saved = []
        for item in result.items:
            label = (item.label or item.path.stem).lower()
            if label not in wanted and item.label not in wanted:
                continue
            art = save_image_artifact(ctx, item.path, label=label, view=label)
            saved.append(art.name)
        if not saved:
            for item in result.items:
                art = save_image_artifact(
                    ctx, item.path, label=item.label or item.path.stem, view=item.label or ""
                )
                saved.append(art.name)
        return f"Generated views: {', '.join(saved)}. knobs={echo}"
