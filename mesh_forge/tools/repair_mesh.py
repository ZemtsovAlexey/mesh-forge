from __future__ import annotations

from pydantic_ai import RunContext

from mesh_forge.agent.deps import ChatDeps
from mesh_forge.ops.geometry import keep_largest_component, load_mesh, remove_needle_faces, try_make_watertight
from mesh_forge.ops.repair import repair_with_pymeshlab
from mesh_forge.tools.base import MeshTool
from mesh_forge.tools.common import resolve_mesh, save_mesh_artifact


class RepairMesh(MeshTool):
    title = "Ремонт"
    heavy = True

    def run(
        self,
        ctx: RunContext[ChatDeps],
        mesh_ref: str | None = None,
        close_holes: bool = True,
        keep_largest: bool = True,
        remove_needles: bool = True,
    ) -> str:
        """Repair mesh: holes, non-manifold, drop floaters, needle faces. Uses current mesh if mesh_ref omitted."""
        src = resolve_mesh(ctx, mesh_ref)
        mesh = load_mesh(src)
        notes: list[str] = []
        if keep_largest:
            mesh = keep_largest_component(mesh, single=True)
            notes.append("largest-component")
        if remove_needles:
            mesh = remove_needle_faces(mesh)
            notes.append("needles")
        if close_holes:
            try:
                mesh = repair_with_pymeshlab(mesh, smooth_iters=0)
                notes.append("pymeshlab")
            except Exception:
                mesh = try_make_watertight(mesh)
                notes.append("watertight-fallback")
        art = save_mesh_artifact(ctx, mesh, "repaired.stl", label="repaired")
        return f"Repaired {src.name} → {art.name} ({', '.join(notes) or 'no-op'}). Faces={len(mesh.faces)}"
