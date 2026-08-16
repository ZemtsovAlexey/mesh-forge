from __future__ import annotations

from pydantic_ai import RunContext

from mesh_forge.agent.deps import ChatDeps
from mesh_forge.ops.geometry import load_mesh, orient_upright
from mesh_forge.tools.base import MeshTool
from mesh_forge.tools.common import resolve_mesh, save_mesh_artifact


class OrientMesh(MeshTool):
    title = "Ориентация"

    def run(self, ctx: RunContext[ChatDeps], mesh_ref: str | None = None) -> str:
        """Seat the mesh upright on the ground (stable base, +Y up). Uses current mesh if mesh_ref omitted."""
        src = resolve_mesh(ctx, mesh_ref)
        mesh = orient_upright(load_mesh(src))
        art = save_mesh_artifact(ctx, mesh, "oriented.stl", label="oriented")
        return (
            f"Oriented {src.name} → {art.name}. "
            "look(target='mesh'). Если ориентация хуже — restore_mesh(to='previous')."
        )
