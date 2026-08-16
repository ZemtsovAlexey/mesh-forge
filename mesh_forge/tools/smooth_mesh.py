from __future__ import annotations

from pydantic_ai import RunContext

from mesh_forge.agent.deps import ChatDeps
from mesh_forge.ops.geometry import load_mesh, smooth_mesh as _smooth
from mesh_forge.tools.base import MeshTool
from mesh_forge.tools.common import resolve_mesh, save_mesh_artifact


class SmoothMesh(MeshTool):
    title = "Сглаживание"

    def run(self, ctx: RunContext[ChatDeps], mesh_ref: str | None = None, iterations: int = 2) -> str:
        """Laplacian smooth. iterations 1–5. Uses current mesh if mesh_ref omitted."""
        src = resolve_mesh(ctx, mesh_ref)
        mesh = _smooth(load_mesh(src), iterations=iterations)
        art = save_mesh_artifact(ctx, mesh, "smoothed.stl", label="smoothed")
        return (
            f"Smoothed {src.name} → {art.name} (iterations={max(0, min(iterations, 5))}). "
            "look(target='mesh'). Если пики или каша — restore_mesh(to='previous')."
        )
