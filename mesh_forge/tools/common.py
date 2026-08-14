from __future__ import annotations

from pathlib import Path

from pydantic_ai import RunContext

from mesh_forge.agent.deps import ChatDeps
from mesh_forge.chat.models import Artifact
from mesh_forge.ops.geometry import load_mesh, save_mesh


def resolve_mesh(ctx: RunContext[ChatDeps], mesh_ref: str | None = None) -> Path:
    store = ctx.deps.store
    chat_id = ctx.deps.chat_id
    if mesh_ref and mesh_ref.strip():
        return store.resolve_ref(chat_id, mesh_ref.strip())
    current = store.current_mesh(chat_id)
    if current is None:
        raise FileNotFoundError("Нет текущего mesh. Сгенерируйте или прикрепите STL.")
    return current


def save_mesh_artifact(
    ctx: RunContext[ChatDeps],
    mesh,
    filename: str,
    *,
    label: str = "",
    tool_id: str = "",
) -> Artifact:
    dest = ctx.deps.store.new_file(ctx.deps.chat_id, filename)
    save_mesh(mesh, dest)
    ctx.deps.store.set_current_mesh(ctx.deps.chat_id, dest)
    art = ctx.deps.store.artifact_from_path(ctx.deps.chat_id, dest, label=label or dest.stem)
    ctx.deps.emit_artifact(art, tool_id=tool_id)
    return art


def save_image_artifact(
    ctx: RunContext[ChatDeps],
    src: Path,
    *,
    label: str = "",
    view: str = "",
    tool_id: str = "",
) -> Artifact:
    dest = ctx.deps.store.new_file(ctx.deps.chat_id, f"{view or src.stem}{src.suffix or '.png'}")
    dest.write_bytes(src.read_bytes())
    art = ctx.deps.store.artifact_from_path(
        ctx.deps.chat_id, dest, label=label or view or dest.stem, view=view
    )
    ctx.deps.emit_artifact(art, tool_id=tool_id)
    return art


def load_current_or_ref(ctx: RunContext[ChatDeps], mesh_ref: str | None = None):
    return load_mesh(resolve_mesh(ctx, mesh_ref))
