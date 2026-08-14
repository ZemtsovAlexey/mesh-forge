from __future__ import annotations

from pydantic_ai import RunContext

from mesh_forge.agent.deps import ChatDeps
from mesh_forge.tools.base import MeshTool


class ListArtifacts(MeshTool):
    title = "Файлы"

    def run(self, ctx: RunContext[ChatDeps]) -> str:
        """List files in this chat: images, meshes, attachments. Use before images_to_mesh if you need refs."""
        arts = ctx.deps.store.list_files(ctx.deps.chat_id)
        current = ctx.deps.store.current_mesh(ctx.deps.chat_id)
        attached = ctx.deps.attachments
        lines = []
        if attached:
            lines.append("This-turn attachments:")
            for art in attached:
                lines.append(f"- {art.id} ({art.kind}) {art.label}")
        if current:
            lines.append(f"Current mesh: {current.name}")
        if arts:
            lines.append("Files:")
            for art in arts:
                extra = f" view={art.view}" if art.view else ""
                lines.append(f"- {art.id} ({art.kind}) {art.label}{extra}")
        if not lines:
            return "No files in this chat yet."
        return "\n".join(lines)
