from __future__ import annotations

from pydantic_ai import RunContext

from mesh_forge.agent.deps import ChatDeps
from mesh_forge.mesh_qc import analyze_mesh
from mesh_forge.ops.geometry import load_mesh
from mesh_forge.tools.base import MeshTool
from mesh_forge.tools.common import resolve_mesh


class InspectMesh(MeshTool):
    title = "Осмотр"

    def run(self, ctx: RunContext[ChatDeps], mesh_ref: str | None = None) -> str:
        """Inspect a mesh: bbox mm, faces, watertight, components. Omit mesh_ref to use the current mesh."""
        path = resolve_mesh(ctx, mesh_ref)
        stats = analyze_mesh(path)
        extra = ""
        try:
            mesh = load_mesh(path)
            bodies = 1
            try:
                import numpy as np
                import trimesh

                if len(getattr(mesh, "face_adjacency", [])) > 0:
                    comps = list(
                        trimesh.graph.connected_components(
                            mesh.face_adjacency,
                            min_len=1,
                            nodes=np.arange(len(mesh.faces)),
                        )
                    )
                    bodies = max(1, len(comps))
            except Exception:
                pass
            extra = f"\nComponents: {bodies}"
        except Exception:
            pass
        return f"{path.name}\n{stats.summary()}{extra}"
