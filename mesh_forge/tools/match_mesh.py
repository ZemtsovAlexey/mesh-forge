from __future__ import annotations

from typing import Literal

from pydantic_ai import RunContext

from mesh_forge.agent.deps import ChatDeps
from mesh_forge.ops.edit import EditError, match_in_region
from mesh_forge.ops.geometry import load_mesh
from mesh_forge.ops.region import RegionError
from mesh_forge.tools.base import MeshTool
from mesh_forge.tools.common import LOOK_AFTER, resolve_edit_target, resolve_mesh, save_mesh_artifact

_Region = Literal["legs", "seat", "back", "left", "right", "top", "bottom", "front"]
_How = Literal["mirror", "height", "flat"]


class MatchMesh(MeshTool):
    title = "Выровнять"
    heavy = True

    def run(
        self,
        ctx: RunContext[ChatDeps],
        how: _How = "mirror",
        region: _Region | None = None,
        mesh_ref: str | None = None,
    ) -> str:
        """Fix part geometry in region. how=mirror copy the better side; height equalize; flat make a plane.

        region: legs|seat|back|left|right|top|bottom|front. Omit if the user clicked the mesh.
        Not for extra volume (remove_mesh) or noise (smooth_mesh).
        """
        src = resolve_mesh(ctx, mesh_ref)
        try:
            label, box, _ = resolve_edit_target(ctx, region)
            mesh, stats = match_in_region(load_mesh(src), label, how, box=box)
        except (RegionError, EditError) as exc:
            return f"match_mesh skipped: {exc} Click the mesh or pass region."
        except Exception as exc:
            return f"match_mesh failed: {exc}. restore_mesh(to='previous')."
        art = save_mesh_artifact(ctx, mesh, "matched.stl", label="matched")
        extra = stats.get("donor") or stats.get("parts") or ""
        return (
            f"Matched {stats.get('region', label)} how={how} on {src.name} → {art.name} {extra}. "
            f"{LOOK_AFTER}"
        )
