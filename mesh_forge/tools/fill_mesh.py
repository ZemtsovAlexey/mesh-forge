from __future__ import annotations

from typing import Literal

from pydantic_ai import RunContext

from mesh_forge.agent.deps import ChatDeps
from mesh_forge.ops.edit import EditError, fill_in_region
from mesh_forge.ops.geometry import load_mesh
from mesh_forge.ops.region import RegionError
from mesh_forge.tools.base import MeshTool
from mesh_forge.tools.common import LOOK_AFTER, resolve_edit_target, resolve_mesh, save_mesh_artifact

_Region = Literal["legs", "seat", "back", "left", "right", "top", "bottom", "front"]


class FillMesh(MeshTool):
    title = "Заделать"
    heavy = True

    def run(
        self,
        ctx: RunContext[ChatDeps],
        region: _Region | None = None,
        mesh_ref: str | None = None,
    ) -> str:
        """Close a hole or gap on one surface in region.

        region: legs|seat|back|left|right|top|bottom|front. Omit if the user clicked the mesh.
        Extra blob → remove_mesh. Two separate bodies → join_mesh.
        """
        src = resolve_mesh(ctx, mesh_ref)
        try:
            label, box, _ = resolve_edit_target(ctx, region)
            mesh, stats = fill_in_region(load_mesh(src), label, box=box)
        except (RegionError, EditError) as exc:
            return f"fill_mesh skipped: {exc} Click the mesh or pass region."
        except Exception as exc:
            return f"fill_mesh failed: {exc}. restore_mesh(to='previous')."
        art = save_mesh_artifact(ctx, mesh, "filled.stl", label="filled")
        return (
            f"Filled {stats.get('region', label)} on {src.name} → {art.name} "
            f"(+{stats['faces_added']} faces). {LOOK_AFTER}"
        )
