from __future__ import annotations

from typing import Literal

from pydantic_ai import RunContext

from mesh_forge.agent.deps import ChatDeps
from mesh_forge.ops.edit import EditError, offset_in_region
from mesh_forge.ops.geometry import load_mesh
from mesh_forge.ops.region import RegionError
from mesh_forge.tools.base import MeshTool
from mesh_forge.tools.common import LOOK_AFTER, resolve_edit_target, resolve_mesh, save_mesh_artifact

_Region = Literal["legs", "seat", "back", "left", "right", "top", "bottom", "front"]
_Knife = Literal["left", "right", "top", "bottom", "front", "back"]


class OffsetMesh(MeshTool):
    title = "Толщина"
    heavy = True

    def run(
        self,
        ctx: RunContext[ChatDeps],
        amount: float = 0.04,
        region: _Region | None = None,
        knife: _Knife | None = None,
        along: _Knife | None = None,
        mesh_ref: str | None = None,
    ) -> str:
        """Inflate (+) or dent (−) along normals in the region. amount −0.25..0.25 of size.

        Not for extra blobs (remove_mesh) or holes (fill_mesh).
        knife=side is mesh AABB, not look camera.
        """
        src = resolve_mesh(ctx, mesh_ref)
        try:
            _, pick = ctx.deps.store.active_mesh_target(ctx.deps.chat_id)
            has_pick = len(pick) >= 3
            if knife and not has_pick:
                mesh, stats = offset_in_region(
                    load_mesh(src), knife, amount, knife=knife, along=along or ""
                )
            else:
                label, box, protect = resolve_edit_target(ctx, region)
                mesh, stats = offset_in_region(
                    load_mesh(src),
                    label,
                    amount,
                    box=box,
                    pick=pick if (not protect and has_pick) else None,
                )
        except (RegionError, EditError) as exc:
            return f"offset_mesh skipped: {exc} Click or pass region."
        except Exception as extra:
            return f"offset_mesh failed: {extra}. restore_mesh(to='previous')."
        art = save_mesh_artifact(ctx, mesh, "offset.stl", label="offset")
        return (
            f"Offset {stats.get('region')} amount={stats['amount']} on {src.name} "
            f"→ {art.name}. {LOOK_AFTER}"
        )
