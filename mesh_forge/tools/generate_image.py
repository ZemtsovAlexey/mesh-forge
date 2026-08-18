from __future__ import annotations

from pydantic_ai import RunContext

from mesh_forge.adapters import ComfyUiClient
from mesh_forge.agent.deps import ChatDeps
from mesh_forge.tools.base import MeshTool
from mesh_forge.tools.common import save_image_artifact
from mesh_forge.tools.knobs import ImageKnobs, ViewName, apply_image_knobs


class GenerateImage(MeshTool):
    title = "Изображение"
    heavy = True

    def run(
        self,
        ctx: RunContext[ChatDeps],
        prompt: str,
        view: ViewName = "front",
        seed: int | None = None,
        quality: str | None = None,
        steps: int | None = None,
        cfg: float | None = None,
        style: str | None = None,
    ) -> str:
        """Generate one view image from a text prompt (ComfyUI).

        Knobs (all optional; omit = config defaults):
        - seed: new seed to redo; omit = random
        - quality: draft (fast turbo) or quality (cleaner, slower)
        - steps, cfg: sampler
        - style: clay (default, best for mesh) or color
        Redo → new seed. style=color only if the user asked for a colored/textured picture.
        Material words (wood, metal) are not a reason to leave clay.
        Prompt: English subject only; isolation/framing/camera is added automatically.
        """
        from mesh_forge import progress as prog

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
        prog.start(ctx.deps.chat_id, "generate_image", "image")
        views = client.generate_front(
            prompt,
            work,
            project_id=ctx.deps.chat_id,
            seed=echo["seed"],
        )
        art = None
        for item in views.items:
            art = save_image_artifact(
                ctx, item.path, label=item.label or view, view=item.label or view
            )
        if art is None:
            return f"No image produced. knobs={echo}"
        return f"Generated {view} image {art.name}. knobs={echo}"
