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
    expose = False

    def run(
        self,
        ctx: RunContext[ChatDeps],
        mesh_ref: str | None = None,
        close_holes: bool = True,
        keep_largest: bool = False,
        remove_needles: bool = True,
    ) -> str:
        """Repair mesh: holes, non-manifold, optional floaters, needle faces. Uses current mesh if mesh_ref omitted.

        Open/non-watertight Hunyuan output is normal.
        keep_largest: only if there are obvious separate floaters (default false).
        If this repair makes the shape worse, restore_mesh.
        """
        from mesh_forge.mesh_qc import mesh_is_usable

        src = resolve_mesh(ctx, mesh_ref)
        ok, qc = mesh_is_usable(src, min_vertices=32, min_faces=16)
        if not ok:
            return (
                f"Cannot repair {src.name}: mesh is empty or too broken.\n{qc}\n"
                "restore_mesh(to='source' or 'previous')."
            )
        try:
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
            return (
                f"Repaired {src.name} → {art.name} ({', '.join(notes) or 'no-op'}). Faces={len(mesh.faces)}. "
                "look(target='mesh'). Если форма хуже — restore_mesh(to='previous')."
            )
        except Exception as exc:
            return (
                f"Repair failed on {src.name}: {exc}. "
                "restore_mesh(to='previous' or 'source')."
            )
