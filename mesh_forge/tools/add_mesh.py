from __future__ import annotations

from typing import Literal

from pydantic_ai import RunContext

from mesh_forge.agent.deps import ChatDeps
from mesh_forge.ops.edit import EditError, add_primitive_in_region
from mesh_forge.ops.geometry import load_mesh
from mesh_forge.ops.region import RegionError
from mesh_forge.tools.base import MeshTool
from mesh_forge.tools.common import LOOK_AFTER, resolve_edit_target, resolve_mesh, save_mesh_artifact

_Region = Literal["legs", "seat", "back", "left", "right", "top", "bottom", "front"]
_Shape = Literal["cylinder", "box", "sphere"]


class AddMesh(MeshTool):
    title = "Добавить"
    heavy = True

    def run(
        self,
        ctx: RunContext[ChatDeps],
        shape: _Shape = "cylinder",
        region: _Region | None = None,
        mesh_ref: str | None = None,
    ) -> str:
        """Add a cylinder, box, or sphere in the region (or at a click) and weld it on.

        For a missing simple part (leg, knob). Not a photo rebuild (images_to_mesh).
        """
        src = resolve_mesh(ctx, mesh_ref)
        try:
            label, box, protect = resolve_edit_target(ctx, region)
            _, pick = ctx.deps.store.active_mesh_target(ctx.deps.chat_id)
            mesh, stats = add_primitive_in_region(
                load_mesh(src),
                label,
                shape,
                box=box,
                pick=pick if (not protect and len(pick) >= 3) else None,
            )
        except (RegionError, EditError) as exc:
            return f"add_mesh skipped: {exc} Click or pass region."
        except Exception as extra:
            return f"add_mesh failed: {extra}. restore_mesh(to='previous')."
        art = save_mesh_artifact(ctx, mesh, "added.stl", label="added")
        return (
            f"Added {stats['shape']} in {stats.get('region')} on {src.name} "
            f"→ {art.name} (+{stats['faces_added']} faces). {LOOK_AFTER}"
        )
