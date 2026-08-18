from __future__ import annotations

from pydantic_ai import RunContext

from mesh_forge.agent.deps import ChatDeps
from mesh_forge.ops.edit import remesh_mesh
from mesh_forge.ops.geometry import load_mesh
from mesh_forge.tools.base import MeshTool
from mesh_forge.tools.common import LOOK_AFTER, resolve_mesh, save_mesh_artifact


class RemeshMesh(MeshTool):
    title = "Пересетка"
    heavy = True

    def run(
        self,
        ctx: RunContext[ChatDeps],
        mesh_ref: str | None = None,
        target_faces: int | None = None,
    ) -> str:
        """Reduce face count. Default: half of current (min 1000). Does not change the intended shape.

        Too many triangles / simplify mesh. Not for holes, extra blobs, or symmetry.
        """
        src = resolve_mesh(ctx, mesh_ref)
        mesh = load_mesh(src)
        n = int(len(mesh.faces))
        mesh = remesh_mesh(mesh, target_faces)
        art = save_mesh_artifact(ctx, mesh, "remeshed.stl", label="remeshed")
        return (
            f"Remeshed {src.name} → {art.name} ({n} → {len(mesh.faces)} faces). {LOOK_AFTER}"
        )
