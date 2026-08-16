from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

from api.deps import get_runner, get_store
from api.schemas import ChatDetail, ChatSummaryOut, CreateChatRequest, MessageOut, RenameChatRequest
from mesh_forge.agent.runner import request_stop
from mesh_forge.chat.models import Artifact

router = APIRouter(prefix="/api/chats", tags=["chats"])

_IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp"}
_MESH_EXT = {".stl", ".obj", ".glb", ".gltf"}


def _hydrate_tools(messages: list) -> None:
    from mesh_forge.tools.base import tool_stage_label, tool_title

    for message in messages:
        for tool in message.tools:
            tool.title = tool.title or tool_title(tool.name)
            if tool.stage:
                tool.stage = tool_stage_label(tool.name, tool.stage)


def _detail(chat_id: str) -> ChatDetail:
    store = get_store()
    meta = store.get_meta(chat_id)
    messages = store.load_messages(chat_id)
    _hydrate_tools(messages)
    return ChatDetail(
        id=meta.id,
        title=meta.title,
        created_at=meta.created_at,
        updated_at=meta.updated_at,
        current_mesh=meta.current_mesh,
        messages=[MessageOut.model_validate(m.model_dump()) for m in messages],
    )


@router.get("", response_model=list[ChatSummaryOut])
def list_chats() -> list[ChatSummaryOut]:
    return [ChatSummaryOut.model_validate(c.model_dump()) for c in get_store().list_chats()]


@router.post("", response_model=ChatDetail, status_code=201)
def create_chat(body: CreateChatRequest | None = None) -> ChatDetail:
    title = body.title if body else "Новый чат"
    meta = get_store().create_chat(title)
    return _detail(meta.id)


@router.get("/{chat_id}", response_model=ChatDetail)
def get_chat(chat_id: str) -> ChatDetail:
    try:
        return _detail(chat_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.patch("/{chat_id}", response_model=ChatDetail)
def rename_chat(chat_id: str, body: RenameChatRequest) -> ChatDetail:
    try:
        get_store().rename(chat_id, body.title)
        return _detail(chat_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.delete("/{chat_id}", status_code=204)
def delete_chat(chat_id: str) -> None:
    try:
        get_store().get_meta(chat_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    request_stop(chat_id)
    get_store().delete(chat_id)


@router.get("/{chat_id}/files/{name}")
def get_file(chat_id: str, name: str):
    try:
        path = get_store().resolve_file(chat_id, name)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    if not path.is_file():
        raise HTTPException(404, "File not found")
    suffix = path.suffix.lower()
    media = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".stl": "model/stl",
        ".obj": "text/plain",
        ".glb": "model/gltf-binary",
    }.get(suffix, "application/octet-stream")
    return FileResponse(path, media_type=media, filename=path.name)


@router.post("/{chat_id}/stop")
def stop_chat(chat_id: str) -> dict:
    try:
        get_store().get_meta(chat_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    request_stop(chat_id)
    return {"ok": True}


@router.post("/{chat_id}/messages")
async def post_message(
    chat_id: str,
    text: str = Form(""),
    files: list[UploadFile] | None = File(None),
    reply_to: str = Form(""),
    reply_artifacts: str = Form(""),
):
    store = get_store()
    try:
        store.get_meta(chat_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc

    attachments: list[Artifact] = []
    for upload in files or []:
        if not upload or not upload.filename:
            continue
        suffix = Path(upload.filename).suffix.lower() or ".bin"
        data = await upload.read()
        dest = store.save_bytes(chat_id, upload.filename, data)
        if suffix in _MESH_EXT:
            store.set_current_mesh(chat_id, dest, role="source")
        attachments.append(store.artifact_from_path(chat_id, dest, label=upload.filename))

    reply_ids = [part.strip() for part in (reply_artifacts or "").replace(";", ",").split(",") if part.strip()]
    runner = get_runner()

    async def events():
        async for chunk in runner.stream_turn(
            chat_id,
            text,
            attachments,
            reply_to=reply_to,
            reply_artifact_ids=reply_ids,
        ):
            yield chunk

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
