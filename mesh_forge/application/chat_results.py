from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any
from urllib.parse import quote

from mesh_forge.application.pipeline_run import abs_from_rel, rel_to_project
from mesh_forge.application.prompt_chat import (
    ChatArtifact,
    ChatMessage,
    PromptChatService,
    _new_message_id,
)
from mesh_forge.manifest import ProjectManifest

logger = logging.getLogger("mesh_forge.chat_results")


def media_url(project_id: str, rel_path: str) -> str:
    return f"/api/projects/{project_id}/media/{quote(rel_path.replace(chr(92), '/'), safe='/')}"


def enrich_message_artifacts(manifest: ProjectManifest, message: dict[str, Any]) -> dict[str, Any]:
    arts = message.get("artifacts") or []
    out = []
    for item in arts:
        if not isinstance(item, dict):
            continue
        row = dict(item)
        path = str(row.get("path") or "").replace("\\", "/")
        if path:
            row["url"] = media_url(manifest.id, path)
        out.append(row)
    message["artifacts"] = out
    return message


def enrich_chat_payload(manifest: ProjectManifest, data: dict[str, Any]) -> dict[str, Any]:
    messages = []
    for item in data.get("messages") or []:
        if isinstance(item, dict):
            messages.append(enrich_message_artifacts(manifest, dict(item)))
    data["messages"] = messages
    return data


def resolve_media_path(manifest: ProjectManifest, rel_path: str) -> Path:
    cleaned = (rel_path or "").replace("\\", "/").lstrip("/")
    if not cleaned or ".." in cleaned.split("/"):
        raise FileNotFoundError("Invalid media path")
    path = (manifest.root / cleaned).resolve()
    root = manifest.root.resolve()
    if not str(path).startswith(str(root)) or not path.is_file():
        raise FileNotFoundError(cleaned)
    return path


def _copy_into_media(manifest: ProjectManifest, msg_id: str, src: Path, filename: str) -> str:
    dest_dir = manifest.root / "work" / "chat_media" / msg_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / filename
    shutil.copy2(src, dest)
    return rel_to_project(manifest, dest)


def post_result_message(
    manifest: ProjectManifest,
    content: str,
    *,
    images: list[tuple[str, Path, str]] | None = None,
    mesh_path: Path | None = None,
    kind: str = "result",
    ref_ids: list[str] | None = None,
) -> ChatMessage:
    """Append an assistant result message; copy files so later redos cannot overwrite chat history."""
    msg_id = _new_message_id()
    artifacts: list[ChatArtifact] = []
    for label, src, stage in images or []:
        if not src or not Path(src).is_file():
            continue
        suffix = Path(src).suffix or ".png"
        rel = _copy_into_media(manifest, msg_id, Path(src), f"{label}{suffix}")
        artifacts.append(ChatArtifact(kind="image", label=label, path=rel, stage=stage))

    if mesh_path and Path(mesh_path).is_file():
        try:
            from mesh_forge.render import render_mesh_preview

            preview = manifest.root / "work" / "chat_media" / msg_id / "mesh_preview.png"
            preview.parent.mkdir(parents=True, exist_ok=True)
            render_mesh_preview(Path(mesh_path), preview)
            artifacts.append(
                ChatArtifact(
                    kind="mesh_preview",
                    label="preview",
                    path=rel_to_project(manifest, preview),
                    stage="mesh",
                )
            )
        except Exception as exc:
            logger.warning("Mesh preview failed: %s", exc)
        suffix = Path(mesh_path).suffix or ".stl"
        rel_mesh = _copy_into_media(manifest, msg_id, Path(mesh_path), f"mesh{suffix}")
        artifacts.append(ChatArtifact(kind="mesh", label="mesh", path=rel_mesh, stage="mesh"))

    body = (content or "").strip()

    message = ChatMessage(
        id=msg_id,
        role="assistant",
        content=body,
        kind=kind,
        ref_ids=list(ref_ids or []),
        artifacts=artifacts,
    )
    chat = PromptChatService()
    state = chat.load(manifest)
    state.messages.append(message)
    state.assistant_message = body
    # Result consumed — clear ready so UI doesn't show a fake confirm card.
    if kind in {"result", "front", "views", "photo", "mesh", "edit"}:
        state.ready = False
        if kind == "mesh":
            state.status = "done"
        elif state.status == "ready":
            state.status = "pipeline"
    chat.save(manifest, state)
    return message


def post_pipeline_chat_result(manifest: ProjectManifest, pipe_state: Any, text: str) -> ChatMessage | None:
    """Publish current pipeline gate as a durable chat message with archived images/mesh."""
    step = str(getattr(pipe_state, "step", "") or "")
    status = str(getattr(pipe_state, "status", "") or "")
    if step in {"", "idle"}:
        return None

    images: list[tuple[str, Path, str]] = []
    mesh_path: Path | None = None
    kind = "result"

    raw_images = list(getattr(pipe_state, "images", []) or [])
    if step == "front":
        kind = "front"
        for img in raw_images:
            if img.label == "front" or img.stage == "front":
                path = abs_from_rel(manifest, img.path)
                images.append((img.label, path, img.stage))
                break
    elif step == "views":
        kind = "views"
        seen: set[str] = set()
        for img in raw_images:
            if img.stage != "views":
                continue
            if img.label in seen:
                continue
            seen.add(img.label)
            images.append((img.label, abs_from_rel(manifest, img.path), img.stage))
    elif step == "photo":
        kind = "photo"
        for img in raw_images:
            if img.label in {"preview", "photo"}:
                images.append((img.label, abs_from_rel(manifest, img.path), img.stage))
    elif step == "done":
        kind = "mesh"
        for img in raw_images:
            if img.stage in {"views", "photo"} or img.label in {"preview", "front", "left", "back", "right"}:
                if img.label in {i[0] for i in images}:
                    continue
                images.append((img.label, abs_from_rel(manifest, img.path), img.stage))
        mesh_path = manifest.current_mesh_path()
    else:
        for img in raw_images:
            images.append((img.label, abs_from_rel(manifest, img.path), img.stage))

    if status == "error" and not images and mesh_path is None:
        kind = "text"

    hint = ""
    if step == "front" and status == "ready":
        hint = "\nНапишите «дальше» для проекций или опишите правку — переделаю front."
    elif step == "views" and status == "ready":
        hint = "\nНапишите «дальше» для mesh или правку — переделаю проекции."
    elif step == "photo" and status == "ready":
        hint = "\nНапишите «дальше» для mesh или пришлите другие фото."
    elif step == "done":
        hint = "\nПревью mesh ниже. Можно править в чате или скачать STL."

    body = (text or "").strip() + hint
    return post_result_message(
        manifest,
        body,
        images=images,
        mesh_path=mesh_path,
        kind=kind,
    )
