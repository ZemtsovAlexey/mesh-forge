from __future__ import annotations

from pydantic_ai import RunContext

from mesh_forge.agent.deps import ChatDeps
from mesh_forge.ops.geometry import decimate, load_mesh
from mesh_forge.tools.base import MeshTool
from mesh_forge.tools.common import resolve_mesh, save_mesh_artifact


class DecimateMesh(MeshTool):
    title = "Упрощение"
    heavy = True
    expose = False

    def run(
        self,
        ctx: RunContext[ChatDeps],
        mesh_ref: str | None = None,
        target_faces: int | None = None,
    ) -> str:
        """Reduce face count. target_faces defaults to half of current (min 1000)."""
        src = resolve_mesh(ctx, mesh_ref)
        mesh = load_mesh(src)
        n = int(len(mesh.faces))
        target = int(target_faces) if target_faces else max(1000, n // 2)
        mesh = decimate(mesh, target)
        art = save_mesh_artifact(ctx, mesh, "decimated.stl", label="decimated")
        return (
            f"Decimated {src.name} → {art.name} ({n} → {len(mesh.faces)} faces). "
            "look(target='mesh'). Если дыры или каша — restore_mesh(to='previous')."
        )
