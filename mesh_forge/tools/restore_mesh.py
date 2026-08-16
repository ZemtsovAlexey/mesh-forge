from __future__ import annotations

from typing import Literal

from pydantic_ai import RunContext

from mesh_forge.agent.deps import ChatDeps
from mesh_forge.tools.base import MeshTool


class RestoreMesh(MeshTool):
    title = "Откат"

    def run(
        self,
        ctx: RunContext[ChatDeps],
        to: Literal["previous", "source"] = "previous",
    ) -> str:
        """Restore the working mesh. previous = before the last edit; source = last Hunyuan/upload.

        Use this when a repair/smooth/decimate/orient/scale made the shape worse.
        Do not generate_image or images_to_mesh to undo a bad edit.
        """
        try:
            path = ctx.deps.store.restore_mesh(ctx.deps.chat_id, to)
        except FileNotFoundError as exc:
            return (
                f"{exc} Нельзя откатиться — нет previous/source. "
                "Не вызывай generate_image, пока пользователь не попросит переделать картинку."
            )
        meta = ctx.deps.store.get_meta(ctx.deps.chat_id)
        art = ctx.deps.store.artifact_from_path(ctx.deps.chat_id, path, label=to)
        ctx.deps.emit_artifact(art)
        return (
            f"Restored current mesh → {path.name} ({to}). "
            f"source={meta.source_mesh or '—'} previous={meta.previous_mesh or '—'}. "
            "Продолжай с этим STL. Не generate_image / images_to_mesh."
        )
