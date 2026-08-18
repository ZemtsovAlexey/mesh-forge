from __future__ import annotations

from typing import Literal

from pydantic_ai import RunContext

from mesh_forge.agent.deps import ChatDeps
from mesh_forge.ops.edit import EditError, extract_in_region
from mesh_forge.ops.geometry import load_mesh
from mesh_forge.ops.region import RegionError
from mesh_forge.ops.topo import TopoError
from mesh_forge.tools.base import MeshTool
from mesh_forge.tools.common import LOOK_AFTER, resolve_edit_target, resolve_mesh, resolve_topo, save_mesh_artifact

_Region = Literal["legs", "seat", "back", "left", "right", "top", "bottom", "front"]
_Knife = Literal["left", "right", "top", "bottom", "front", "back"]
_Elem = Literal["vertex", "edge", "face"]


class ExtractMesh(MeshTool):
    title = "Вынести"
    heavy = True

    def run(
        self,
        ctx: RunContext[ChatDeps],
        region: _Region | None = None,
        knife: _Knife | None = None,
        along: _Knife | None = None,
        elem: _Elem | None = None,
        vertex: int | None = None,
        face: int | None = None,
        edge: str | None = None,
        mesh_ref: str | None = None,
    ) -> str:
        """Pull a piece out as its own STL. Remainder stays current.

        Click (vertex/edge/face) or knife=side on mesh axes (not look camera). Both pieces kept.
        To throw away → remove_mesh.
        """
        src = resolve_mesh(ctx, mesh_ref)
        try:
            loaded = load_mesh(src)
            _, pick = ctx.deps.store.active_mesh_target(ctx.deps.chat_id)
            has_pick = len(pick) >= 3
            topo = resolve_topo(ctx, loaded, elem=elem, vertex=vertex, face=face, edge=edge)
            if topo:
                rest, piece, stats = extract_in_region(loaded, "topo", topo=topo)
            elif knife and not has_pick:
                rest, piece, stats = extract_in_region(
                    loaded, knife, knife=knife, along=along or ""
                )
            else:
                label, box, protect = resolve_edit_target(ctx, region)
                rest, piece, stats = extract_in_region(
                    loaded,
                    label,
                    box=box,
                    pick=pick if (not protect and has_pick) else None,
                )
        except (RegionError, EditError, TopoError) as exc:
            return (
                f"extract_mesh skipped: {exc} Click the piece or knife=side."
            )
        except Exception as extra:
            return f"extract_mesh failed: {extra}. restore_mesh(to='previous')."
        save_mesh_artifact(ctx, rest, "remainder.stl", label="remainder")
        extracted = save_mesh_artifact(
            ctx, piece, "extracted.stl", label="extracted", make_current=False
        )
        where = stats.get("region", "pick")
        return (
            f"Extracted {where} from {src.name} → {extracted.name} "
            f"({stats['faces_extracted']} faces); remainder is current. {LOOK_AFTER}"
        )
