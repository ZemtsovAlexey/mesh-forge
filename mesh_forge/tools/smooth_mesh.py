from __future__ import annotations

from typing import Literal

from pydantic_ai import RunContext

from mesh_forge.agent.deps import ChatDeps
from mesh_forge.ops.edit import EditError, smooth_in_region
from mesh_forge.ops.geometry import load_mesh
from mesh_forge.ops.region import RegionError
from mesh_forge.ops.topo import TopoError, format_topo
from mesh_forge.tools.base import MeshTool
from mesh_forge.tools.common import LOOK_AFTER, resolve_edit_target, resolve_mesh, resolve_topo, save_mesh_artifact

_Region = Literal["legs", "seat", "back", "left", "right", "top", "bottom", "front"]
_Elem = Literal["vertex", "edge", "face"]


class SmoothMesh(MeshTool):
    title = "Сглаживание"

    def run(
        self,
        ctx: RunContext[ChatDeps],
        region: _Region | None = None,
        elem: _Elem | None = None,
        vertex: int | None = None,
        face: int | None = None,
        edge: str | None = None,
        mesh_ref: str | None = None,
        iterations: int = 2,
    ) -> str:
        """Soften noise/small wrinkles. Prefer a region or a clicked vertex/edge/face.

        region: legs|seat|back|left|right|top|bottom|front. Omit if the user clicked the mesh.
        To smooth everything, clear the click first.
        """
        src = resolve_mesh(ctx, mesh_ref)
        label, pick = ctx.deps.store.active_mesh_target(ctx.deps.chat_id)
        try:
            loaded = load_mesh(src)
            topo = resolve_topo(ctx, loaded, elem=elem, vertex=vertex, face=face, edge=edge)
            if topo:
                mesh = smooth_in_region(loaded, "topo", iterations, topo=topo)
                where = format_topo(topo)
            elif region or pick or label:
                name, box, _ = resolve_edit_target(ctx, region)
                mesh = smooth_in_region(loaded, name, iterations, box=box)
                where = name
            else:
                mesh = smooth_in_region(loaded, None, iterations)
                where = "all"
        except (RegionError, EditError, TopoError) as exc:
            return f"smooth_mesh skipped: {exc} look, then retry."
        art = save_mesh_artifact(ctx, mesh, "smoothed.stl", label="smoothed")
        return f"Smoothed {where} on {src.name} → {art.name} (iterations={max(0, min(iterations, 5))}). {LOOK_AFTER}"
