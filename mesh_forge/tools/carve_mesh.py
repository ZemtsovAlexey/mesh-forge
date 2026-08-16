from __future__ import annotations

from typing import Literal

from pydantic_ai import RunContext

from mesh_forge.agent.deps import ChatDeps
from mesh_forge.ops.geometry import CarveError, carve_region, load_mesh, resolve_carve_box
from mesh_forge.tools.base import MeshTool
from mesh_forge.tools.common import resolve_mesh, save_mesh_artifact

_LOOK_AFTER = (
    "look(target='mesh', views='front,left,right', "
    "question='Срезан ли подлокотник, нога или сиденье? Один бок короче? "
    "Ровный срез сбоку — брак. Если да — restore_mesh, не хвали.'). "
    "Если look видит срез — сразу restore_mesh(to='previous'). "
    "Не пиши «отлично»/«целы». Не generate_image."
)


class CarveMesh(MeshTool):
    title = "Вырез"
    heavy = True

    def run(
        self,
        ctx: RunContext[ChatDeps],
        mesh_ref: str | None = None,
        action: Literal["remove", "keep"] = "remove",
        side: str | None = None,
        amount: float = 0.10,
        left: float | None = None,
        right: float | None = None,
        bottom: float | None = None,
        top: float | None = None,
        back: float | None = None,
        front: float | None = None,
    ) -> str:
        """Cut a LOCAL box of extra geometry. Never a full left/right slab — that chops the armrest.

        Axes match look, 0–1 of bbox: X left→right, Y bottom→top, Z back→front.
        Backrest wing: side='right' or 'left', amount 0.08–0.14, AND bottom=0.45, front=0.55.
        Floor blob: side + top=0.35. Always set a second bound (bottom/top or back/front).
        action=remove deletes the box; action=keep crops to it.
        After carve: look front+left+right. If an armrest/leg/seat is gone — restore_mesh. Do not praise.
        """
        src = resolve_mesh(ctx, mesh_ref)
        try:
            box = resolve_carve_box(
                side=side,
                amount=amount,
                left=left,
                right=right,
                bottom=bottom,
                top=top,
                back=back,
                front=front,
            )
            mesh, stats = carve_region(load_mesh(src), box, action=action)
        except CarveError as exc:
            return (
                f"Carve skipped on {src.name}: {exc} "
                "look(target='mesh', views='front,left,right') then retry with a tighter local box. "
                "Не generate_image."
            )
        except Exception as exc:
            return (
                f"Carve failed on {src.name}: {exc}. "
                "restore_mesh(to='previous' or 'source'). Не generate_image."
            )
        art = save_mesh_artifact(ctx, mesh, "carved.stl", label="carved")
        l, r, b, t, bk, f = box
        region = f"box L{l:.2f}–R{r:.2f} B{b:.2f}–T{t:.2f} Bk{bk:.2f}–F{f:.2f}"
        if side:
            region = f"side={side} amount={amount:.2f}, {region}"
        return (
            f"Carved {src.name} → {art.name} ({action} {stats['faces_dropped']} faces, "
            f"{stats['faces_before']} → {stats['faces_after']}; {region}). "
            f"{_LOOK_AFTER}"
        )
