from __future__ import annotations

from pydantic_ai import RunContext

from mesh_forge.agent.deps import ChatDeps
from mesh_forge.ops.edit import EditError
from mesh_forge.ops.geometry import CarveError, load_mesh
from mesh_forge.tools.base import MeshTool
from mesh_forge.tools.common import (
    LOOK_AFTER,
    apply_saved_proposal_mesh,
    carve_painted_mask,
    resolve_mesh,
    save_mesh_artifact,
)


class RemoveMesh(MeshTool):
    title = "Удалить"
    heavy = True

    def run(
        self,
        ctx: RunContext[ChatDeps],
        mesh_ref: str | None = None,
    ) -> str:
        """Delete exactly the red faces from mask_mesh. No hops, no extra crumbs."""
        src = resolve_mesh(ctx, mesh_ref)
        proposal = ctx.deps.store.removal_state(ctx.deps.chat_id)
        if proposal:
            try:
                art = apply_saved_proposal_mesh(
                    ctx,
                    proposal,
                    filename="removed_extra.stl",
                    label="removed",
                    role="edit",
                )
            except Exception:
                art = None
            else:
                ctx.deps.store.clear_removal_state(ctx.deps.chat_id)
                ctx.deps.store.clear_mask_state(ctx.deps.chat_id)
                ctx.deps.store.clear_mesh_target(ctx.deps.chat_id)
                return (
                    f"Applied {proposal.get('strategy') or 'remove'} proposal on {src.name} → {art.name}. "
                    + LOOK_AFTER
                )
        try:
            mesh, stats = carve_painted_mask(ctx, load_mesh(src), src.name)
        except (CarveError, EditError) as exc:
            return (
                "remove_mesh skipped: "
                f"{exc} Paint with mask_mesh first and check the red overlay."
            )
        except Exception as exc:
            return f"remove_mesh failed: {exc}. restore_mesh(to='previous')."
        art = save_mesh_artifact(ctx, mesh, "removed.stl", label="removed")
        n = int(stats.get("faces_dropped") or 0)
        extra = int(stats.get("faces_extra") or 0)
        note = f"Removed painted mask on {src.name} → {art.name} ({n} faces). "
        if extra:
            note += f"Warning: {extra} extra faces vanished. "
        return note + LOOK_AFTER
