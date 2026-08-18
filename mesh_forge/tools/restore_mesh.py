from __future__ import annotations

from typing import Literal

from pydantic_ai import RunContext

from mesh_forge.agent.deps import ChatDeps
from mesh_forge.tools.base import MeshTool
from mesh_forge.tools.common import emit_mesh_preview


class RestoreMesh(MeshTool):
    title = "Откат"

    def run(
        self,
        ctx: RunContext[ChatDeps],
        to: Literal["previous", "source"] = "previous",
    ) -> str:
        """Undo mesh edit. previous=one step back (stack); source=last Hunyuan/upload.

        Shape got worse → this.
        """
        try:
            path = ctx.deps.store.restore_mesh(ctx.deps.chat_id, to)
        except FileNotFoundError as exc:
            return f"{exc} Нельзя откатиться — нет previous/source."
        meta = ctx.deps.store.get_meta(ctx.deps.chat_id)
        art = ctx.deps.store.artifact_from_path(ctx.deps.chat_id, path, label=to)
        ctx.deps.emit_artifact(art)
        emit_mesh_preview(ctx, path)
        return (
            f"Restored current mesh → {path.name} ({to}). "
            f"source={meta.source_mesh or '—'} previous={meta.previous_mesh or '—'} "
            f"history={len(meta.mesh_history or [])}. "
            "Продолжай с этим STL."
        )
