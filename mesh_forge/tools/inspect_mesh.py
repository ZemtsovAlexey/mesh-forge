from __future__ import annotations

from pydantic_ai import RunContext

from mesh_forge.agent.deps import ChatDeps
from mesh_forge.mesh_qc import analyze_mesh
from mesh_forge.ops.geometry import load_mesh
from mesh_forge.tools.base import MeshTool
from mesh_forge.tools.common import resolve_mesh


class InspectMesh(MeshTool):
    title = "Осмотр"
    expose = False

    def run(self, ctx: RunContext[ChatDeps], mesh_ref: str | None = None) -> str:
        """Inspect a mesh: bbox mm, faces, watertight, components. Omit mesh_ref to use the current mesh.

        Informational only. Open/non-watertight and many patches are normal for Hunyuan.
        If a recent edit made the shape worse, restore_mesh.
        """
        path = resolve_mesh(ctx, mesh_ref)
        stats = analyze_mesh(path)
        extra = ""
        bodies = 1
        try:
            mesh = load_mesh(path)
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
            extra = _component_line(bodies)
        except Exception:
            pass
        return f"{path.name}\n{stats.summary()}{extra}\n{_inspect_advice(stats, bodies)}"


def _component_line(bodies: int) -> str:
    if bodies <= 1:
        return "\nКомпоненты: 1 тело"
    if bodies > 32:
        return (
            f"\nЛоскутов: {bodies} (открытая поверхность реконструкции, не {bodies} отдельных объектов)"
        )
    return f"\nКомпоненты: {bodies}"


def _inspect_advice(stats, bodies: int) -> str:
    if stats.vertex_count == 0 or stats.triangle_count == 0:
        return (
            "Пустой меш: restore_mesh(to='source')."
        )
    lines = ["Это осмотр: bbox, грани, watertight, компоненты."]
    if not stats.watertight:
        lines.append("«Не замкнут» для Hunyuan — норма.")
    if bodies > 32:
        lines.append("Много лоскутов — типичная открытая поверхность реконструкции.")
    lines.append("Если недавняя правка испортила форму — restore_mesh(to='previous' или 'source').")
    return "\n".join(lines)
