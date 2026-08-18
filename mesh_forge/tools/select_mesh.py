from __future__ import annotations

from typing import Literal

from pydantic_ai import RunContext

from mesh_forge.agent.deps import ChatDeps
from mesh_forge.ops.geometry import load_mesh
from mesh_forge.ops.topo import TopoError, format_topo, viewport_hit
from mesh_forge.tools.base import MeshTool
from mesh_forge.tools.common import resolve_mesh

_View = Literal["viewer", "front", "left", "right", "back", "top"]
_Elem = Literal["vertex", "edge", "face"]


def inherit_look_camera(
    ctx: RunContext[ChatDeps],
    views: str,
    yaw: float | None,
    pitch: float | None,
    zoom: float,
) -> tuple[str, float | None, float | None, float]:
    last = dict(ctx.deps.store.get_meta(ctx.deps.chat_id).look_view or {})
    if not last:
        return views, yaw, pitch, zoom
    token = str(views or "right").split(",")[0].strip().lower() or "right"
    last_views = str(last.get("views") or "").lower()
    last_tokens = {part.strip() for part in last_views.split(",") if part.strip()}
    named = token in {"viewer", "front", "left", "right", "back", "top"}
    if token not in last_tokens and last_tokens:
        return views, yaw, pitch, zoom
    if not named and yaw is None and last.get("yaw") is not None:
        yaw = float(last["yaw"])
    if not named and pitch is None and last.get("pitch") is not None:
        pitch = float(last["pitch"])
    if abs(float(zoom or 1.0) - 1.0) < 1e-6 and last.get("zoom") is not None:
        zoom = float(last["zoom"] or 1.0)
    return views, yaw, pitch, zoom


class SelectMesh(MeshTool):
    title = "Целюсь"
    heavy = True
    expose = False

    def run(
        self,
        ctx: RunContext[ChatDeps],
        views: _View | str = "right",
        x: float = 0.5,
        y: float = 0.62,
        hops: int = 12,
        elem: _Elem = "face",
        yaw: float | None = None,
        pitch: float | None = None,
        zoom: float = 1.0,
        mesh_ref: str | None = None,
    ) -> str:
        """Aim at the mesh like clicking a look preview. Stores a face group for the next edit.

        views: same camera as the look frame you are aiming on. x,y: 0–1 in that frame, (0,0)=top-left.
        The figure is usually near the center — 0.75 is often empty background.
        hops: grow connected faces from the hit (4–24), not the whole skirt.
        Then call remove_mesh WITHOUT knife.
        """
        src = resolve_mesh(ctx, mesh_ref)
        views, yaw, pitch, zoom = inherit_look_camera(ctx, str(views or "right"), yaw, pitch, zoom)
        camera = str(views or "right").split(",")[0].strip().lower() or "right"
        try:
            topo = viewport_hit(
                load_mesh(src),
                camera=camera,
                views=camera,
                yaw=yaw,
                pitch=pitch,
                x=max(0.0, min(1.0, float(x))),
                y=max(0.0, min(1.0, float(y))),
                zoom=zoom,
                kind=elem,
                hops=hops,
            )
        except TopoError as exc:
            return f"select_mesh missed: {exc} Change views/x/y."
        except Exception as exc:
            return f"select_mesh failed: {exc}"
        ctx.deps.store.apply_mesh_topo(ctx.deps.chat_id, topo, mesh_name=src.name)
        n = int(topo.get("faces") or 0)
        snap = float(topo.get("aim_snap") or 0.0)
        extra = ""
        if snap > 0.08:
            extra = (
                f" Aim ({float(x):.2f},{float(y):.2f}) was off the silhouette "
                f"(snapped {snap:.2f} of the frame); orange marker shows the hit. "
            )
        pick = [float(topo["nx"]), float(topo["ny"]), float(topo["nz"])]
        try:
            from mesh_forge.tools.look import _render_mesh_looks

            _render_mesh_looks(
                ctx,
                src,
                views=camera,
                zoom=zoom,
                region="",
                pick=pick,
                yaw=yaw,
                pitch=pitch,
            )
        except Exception:
            extra += " Preview marker skipped."
        return (
            f"Selected {format_topo(topo)} on {src.name} (~{n} faces, hops={int(topo.get('hops', hops))}). "
            f"{extra}"
            "Next: remove_mesh without knife (local patch)."
        )
