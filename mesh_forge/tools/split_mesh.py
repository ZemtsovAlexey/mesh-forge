from __future__ import annotations

from typing import Literal

from pydantic_ai import RunContext

from mesh_forge.agent.deps import ChatDeps
from mesh_forge.ops.edit import EditError, split_in_region
from mesh_forge.ops.geometry import CarveError, load_mesh
from mesh_forge.ops.region import RegionError
from mesh_forge.tools.base import MeshTool
from mesh_forge.tools.common import LOOK_AFTER, resolve_edit_target, resolve_mesh, save_mesh_artifact

_Region = Literal["legs", "seat", "back", "left", "right", "top", "bottom", "front"]


class SplitMesh(MeshTool):
    title = "Разделить"
    heavy = True

    def run(
        self,
        ctx: RunContext[ChatDeps],
        region: _Region | None = None,
        mesh_ref: str | None = None,
    ) -> str:
        """Cut fused parts in region; both pieces stay.

        region: legs|seat|back|left|right|top|bottom|front. Omit if the user clicked the mesh.
        To throw a piece away → remove_mesh. To glue pieces → join_mesh.
        """
        src = resolve_mesh(ctx, mesh_ref)
        try:
            label, box, protect = resolve_edit_target(ctx, region)
            mesh, stats = split_in_region(
                load_mesh(src), label, box=box, protect_sides=protect
            )
        except (CarveError, RegionError, EditError) as exc:
            return f"split_mesh skipped: {exc} Click the mesh or pass region."
        except Exception as exc:
            return f"split_mesh failed: {exc}. restore_mesh(to='previous')."
        art = save_mesh_artifact(ctx, mesh, "split.stl", label="split")
        return (
            f"Split {stats.get('region', label)} on {src.name} → {art.name} "
            f"({stats['faces_dropped']} faces cut). {LOOK_AFTER}"
        )
