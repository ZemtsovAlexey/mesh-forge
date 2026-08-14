from __future__ import annotations

from typing import Literal

from pydantic_ai import RunContext

from mesh_forge.agent.deps import ChatDeps
from mesh_forge.ops.geometry import load_mesh, normalize_height_mm, scale_axis, scale_uniform
from mesh_forge.tools.base import MeshTool
from mesh_forge.tools.common import resolve_mesh, save_mesh_artifact


class ScaleMesh(MeshTool):
    title = "Масштаб"

    def run(
        self,
        ctx: RunContext[ChatDeps],
        mesh_ref: str | None = None,
        factor: float | None = None,
        height_mm: float | None = None,
        axis: Literal["x", "y", "z"] | None = None,
        axis_mm: float | None = None,
    ) -> str:
        """Scale the current mesh. Prefer height_mm (longest axis) or uniform factor. Optional single-axis axis+axis_mm."""
        src = resolve_mesh(ctx, mesh_ref)
        mesh = load_mesh(src)
        if height_mm is not None and height_mm > 0:
            mesh = normalize_height_mm(mesh, float(height_mm))
            note = f"height={height_mm}mm"
        elif axis and axis_mm is not None and axis_mm > 0:
            mesh = scale_axis(mesh, axis, float(axis_mm))
            note = f"{axis}={axis_mm}mm"
        elif factor is not None and factor > 0:
            mesh = scale_uniform(mesh, float(factor))
            note = f"factor={factor}"
        else:
            return "Specify height_mm, factor, or axis+axis_mm."
        art = save_mesh_artifact(ctx, mesh, "scaled.stl", label="scaled")
        extents = [round(float(x), 1) for x in mesh.extents]
        return f"Scaled {src.name} → {art.name} ({note}). BBox mm ≈ {extents}"
