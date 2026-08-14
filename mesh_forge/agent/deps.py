from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from mesh_forge.chat.models import Artifact
from mesh_forge.chat.store import ChatStore


EmitFn = Callable[[dict[str, Any]], None]


@dataclass
class ChatDeps:
    chat_id: str
    store: ChatStore
    attachments: list[Artifact] = field(default_factory=list)
    emit: EmitFn = field(default=lambda _event: None)
    loop: asyncio.AbstractEventLoop | None = None

    def emit_event(self, event_type: str, **payload: Any) -> None:
        data = {"type": event_type, **payload}
        loop = self.loop
        if loop is not None:
            loop.call_soon_threadsafe(self.emit, data)
        else:
            self.emit(data)

    def emit_artifact(self, artifact: Artifact, *, tool_id: str = "") -> None:
        payload = artifact.model_dump()
        if tool_id:
            payload["tool_id"] = tool_id
        self.emit_event("artifact", artifact=payload)

    def files_dir(self) -> Path:
        return self.store.files_dir(self.chat_id)
