from __future__ import annotations

from typing import Literal

from pydantic_ai import RunContext

from mesh_forge.agent.deps import ChatDeps
from mesh_forge.ops.edit import EditError, join_in_region
from mesh_forge.ops.geometry import load_mesh
from mesh_forge.ops.region import RegionError
from mesh_forge.tools.base import MeshTool
from mesh_forge.tools.common import LOOK_AFTER, resolve_edit_target, resolve_mesh, save_mesh_artifact

_Region = Literal["legs", "seat", "back", "left", "right", "top", "bottom", "front"]


class JoinMesh(MeshTool):
    title = "Соединить"
    heavy = True

    def run(
        self,
        ctx: RunContext[ChatDeps],
        region: _Region | None = None,
        mesh_ref: str | None = None,
    ) -> str:
        """Weld loose parts in region into one body.

        region: legs|seat|back|left|right|top|bottom|front. Omit if the user clicked the mesh.
        Hole on one surface → fill_mesh. Cut apart → split_mesh.
        """
        src = resolve_mesh(ctx, mesh_ref)
        try:
            label, box, _ = resolve_edit_target(ctx, region)
            mesh, stats = join_in_region(load_mesh(src), label, box=box)
        except (RegionError, EditError) as exc:
            return f"join_mesh skipped: {exc} Click the mesh or pass region."
        except Exception as exc:
            return f"join_mesh failed: {exc}. restore_mesh(to='previous')."
        art = save_mesh_artifact(ctx, mesh, "joined.stl", label="joined")
        return f"Joined {stats.get('region', label)} on {src.name} → {art.name}. {LOOK_AFTER}"
