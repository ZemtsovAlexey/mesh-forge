from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from typing import Any

from pydantic_ai import CancellationToken
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartDeltaEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ToolCallPart,
)

from mesh_forge import progress as prog
from mesh_forge.agent.deps import ChatDeps
from mesh_forge.agent.mesh_agent import build_agent
from mesh_forge.agent.workspace import (
    build_workspace_brief,
    cited_artifacts,
    format_reply_prompt,
    is_followup,
    with_workspace,
)
from mesh_forge.chat.models import Artifact, MessageBlock, ToolCallRecord, UiMessage
from mesh_forge.chat.store import ChatStore
from mesh_forge.tools.base import tool_stage_label, tool_title

logger = logging.getLogger("mesh_forge.agent.runner")

_cancel_tokens: dict[str, CancellationToken] = {}
_chat_locks: dict[str, asyncio.Lock] = {}


def _chat_lock(chat_id: str) -> asyncio.Lock:
    lock = _chat_locks.get(chat_id)
    if lock is None:
        lock = asyncio.Lock()
        lock = _chat_locks.setdefault(chat_id, lock)
    return lock


def request_stop(chat_id: str) -> None:
    prog.request_cancel(chat_id)
    token = _cancel_tokens.get(chat_id)
    if token is not None:
        token.cancel()
    try:
        from mesh_forge.adapters import ComfyUiClient

        ComfyUiClient().interrupt()
    except Exception:
        pass
    from mesh_forge.runtime import get_gpu_scheduler

    get_gpu_scheduler().wake()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sse(event: dict[str, Any]) -> str:
    name = event.get("type") or "message"
    return f"event: {name}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"


class ChatRunner:
    def __init__(self, store: ChatStore | None = None) -> None:
        self.store = store or ChatStore()
        self._agent = None

    def agent(self):
        if self._agent is None:
            self._agent = build_agent()
        return self._agent

    def reload_agent(self) -> None:
        self._agent = None

    async def stream_turn(
        self,
        chat_id: str,
        text: str,
        attachments: list[Artifact],
        reply_to: str = "",
        reply_artifact_ids: list[str] | None = None,
    ) -> AsyncIterator[str]:
        lock = _chat_lock(chat_id)
        while lock.locked():
            yield _sse({"type": "status", "stage": "wait_turn"})
            try:
                await asyncio.wait_for(lock.acquire(), timeout=20)
                break
            except TimeoutError:
                continue
        else:
            await lock.acquire()
        try:
            async for chunk in self._stream_turn(
                chat_id,
                text,
                attachments,
                reply_to=reply_to,
                reply_artifact_ids=reply_artifact_ids,
            ):
                yield chunk
        finally:
            lock.release()

    async def _stream_turn(
        self,
        chat_id: str,
        text: str,
        attachments: list[Artifact],
        reply_to: str = "",
        reply_artifact_ids: list[str] | None = None,
    ) -> AsyncIterator[str]:
        self.store.get_meta(chat_id)
        if not is_followup(text):
            self.store.maybe_set_title(chat_id, text)
        messages = self.store.load_messages(chat_id)
        reply_ids = [str(x).strip() for x in (reply_artifact_ids or []) if str(x).strip()]
        cited = cited_artifacts(messages, reply_to, reply_ids)
        user_msg = UiMessage(
            id=uuid.uuid4().hex[:10],
            role="user",
            content=text,
            created_at=_now(),
            attachments=attachments,
            reply_to=(reply_to or "").strip(),
            reply_artifact_ids=reply_ids,
        )
        assistant_msg = UiMessage(
            id=uuid.uuid4().hex[:10],
            role="assistant",
            content="",
            created_at=_now(),
        )
        messages.append(user_msg)
        messages.append(assistant_msg)
        self.store.save_messages(chat_id, messages)

        yield _sse({"type": "user", "message": user_msg.model_dump()})
        yield _sse({"type": "assistant_start", "id": assistant_msg.id})

        bus: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        loop = asyncio.get_running_loop()
        token = CancellationToken()
        _cancel_tokens[chat_id] = token
        prog.start(chat_id, "chat", "agent")

        deps = ChatDeps(
            chat_id=chat_id,
            store=self.store,
            attachments=attachments,
            reply_artifacts=cited,
            emit=bus.put_nowait,
            loop=loop,
        )

        prompt = text.strip() or "(see attached files)"
        if attachments:
            bits = ", ".join(f"{a.kind}:{a.id}" for a in attachments)
            prompt = f"{prompt}\n\nAttached this turn: {bits}"
        if cited:
            prompt = f"{prompt}\n\n{format_reply_prompt(cited, (reply_to or '').strip())}"
        prompt = with_workspace(
            prompt,
            build_workspace_brief(self.store, chat_id, attachments, reply_artifacts=cited),
        )

        history = self.store.load_agent_messages(chat_id)

        async def run_agent() -> None:
            try:
                agent = self.agent()
                async with agent.run_stream_events(
                    prompt,
                    deps=deps,
                    message_history=history or None,
                    cancellation_token=token,
                ) as events:
                    async for event in events:
                        mapped = _map_agent_event(event)
                        if mapped:
                            await bus.put(mapped)
                        if getattr(event, "event_kind", "") == "result" or type(event).__name__ == "AgentRunResultEvent":
                            result = getattr(event, "result", None)
                            if result is not None:
                                try:
                                    all_msgs = result.all_messages()
                                    self.store.save_agent_messages(chat_id, all_msgs)
                                except Exception:
                                    logger.exception("failed to persist agent messages")
                await bus.put({"type": "done"})
            except Exception as exc:
                logger.exception("agent run failed")
                await bus.put({"type": "error", "message": str(exc)})
                await bus.put({"type": "done"})

        task = asyncio.create_task(run_agent())
        tools_by_id: dict[str, ToolCallRecord] = {}

        def persist() -> None:
            try:
                self.store.save_messages(chat_id, messages)
            except Exception:
                logger.exception("failed to persist ui messages")

        try:
            while True:
                try:
                    event = await asyncio.wait_for(bus.get(), timeout=0.45)
                except TimeoutError:
                    state = prog.get(chat_id)
                    if state and state.active:
                        last = assistant_msg.tools[-1] if assistant_msg.tools else None
                        if last is None or last.status == "running":
                            tool_name = last.name if last else state.operation
                            yield _sse(
                                {
                                    "type": "tool_progress",
                                    "percent": state.percent,
                                    "stage": tool_stage_label(tool_name, state.stage),
                                }
                            )
                    continue
                etype = event.get("type")
                dirty = False
                followup: list[dict[str, Any]] = []
                if etype == "tool_start":
                    name = str(event.get("name") or "tool")
                    record = ToolCallRecord(
                        id=event.get("id") or uuid.uuid4().hex[:8],
                        name=name,
                        title=tool_title(name),
                        status="running",
                        args=event.get("args") or {},
                    )
                    tools_by_id[record.id] = record
                    assistant_msg.tools.append(record)
                    _append_tool_block(assistant_msg, record.id)
                    event["id"] = record.id
                    event["title"] = record.title
                    prog.start(chat_id, name, "")
                    dirty = True
                elif etype == "tool_end":
                    rec = tools_by_id.get(event.get("id") or "")
                    if rec is None and assistant_msg.tools:
                        rec = assistant_msg.tools[-1]
                    if rec is not None:
                        rec.status = "error" if event.get("ok") is False else "ok"
                        rec.summary = str(event.get("summary") or "")[:2000]
                        knobs = _extract_knobs(rec.summary)
                        if knobs:
                            rec.knobs = knobs
                            event["knobs"] = knobs
                    dirty = True
                elif etype == "tool_progress":
                    rec = None
                    if assistant_msg.tools:
                        last = assistant_msg.tools[-1]
                        if last.status == "running":
                            rec = last
                    if rec is not None:
                        rec.progress = float(event.get("percent") or rec.progress)
                        rec.stage = tool_stage_label(rec.name, str(event.get("stage") or rec.stage))
                        event["stage"] = rec.stage
                elif etype == "artifact":
                    art = Artifact.model_validate(event["artifact"])
                    tool_id = event["artifact"].get("tool_id") or (
                        assistant_msg.tools[-1].id if assistant_msg.tools else ""
                    )
                    if tool_id:
                        for rec in assistant_msg.tools:
                            if rec.id == tool_id:
                                rec.artifacts.append(art)
                                break
                        else:
                            assistant_msg.artifacts.append(art)
                    else:
                        if assistant_msg.tools:
                            assistant_msg.tools[-1].artifacts.append(art)
                        else:
                            assistant_msg.artifacts.append(art)
                    dirty = True
                elif etype == "text_delta":
                    _append_text(assistant_msg, str(event.get("delta") or ""))
                elif etype == "error":
                    if not assistant_msg.content:
                        assistant_msg.content = str(event.get("message") or "Ошибка")
                    still = [rec for rec in assistant_msg.tools if rec.status == "running"]
                    _fail_running_tools(assistant_msg.tools, str(event.get("message") or "Ошибка"))
                    for rec in still:
                        followup.append(
                            {"type": "tool_end", "id": rec.id, "ok": False, "summary": rec.summary}
                        )
                    dirty = True
                if dirty:
                    persist()
                yield _sse(event)
                for item in followup:
                    yield _sse(item)
                if etype == "done":
                    break
        finally:
            _cancel_tokens.pop(chat_id, None)
            if not task.done():
                task.cancel()
            try:
                await task
            except Exception:
                pass
            still = [rec for rec in assistant_msg.tools if rec.status == "running"]
            _fail_running_tools(assistant_msg.tools, "Прервано")
            for rec in still:
                yield _sse({"type": "tool_end", "id": rec.id, "ok": False, "summary": rec.summary})
            prog.finish(chat_id, ok=True)
            persist()


def _append_text(message: UiMessage, delta: str) -> None:
    if not delta:
        return
    message.content += delta
    if message.blocks and message.blocks[-1].kind == "text":
        message.blocks[-1].text += delta
    else:
        message.blocks.append(MessageBlock(kind="text", text=delta))


def _append_tool_block(message: UiMessage, tool_id: str) -> None:
    message.blocks.append(MessageBlock(kind="tool", tool_id=tool_id))


def _fail_running_tools(tools: list[ToolCallRecord], message: str) -> None:
    note = (message or "Ошибка").strip()[:2000]
    for rec in tools:
        if rec.status == "running":
            rec.status = "error"
            rec.summary = rec.summary or note


def _tool_succeeded(content: str) -> bool:
    text = (content or "").strip()
    if not text:
        return False
    lower = text.lower()
    if lower.startswith("error") or lower.startswith("cannot "):
        return False
    if "input should be" in lower or "'type': 'model_type'" in text or '"type": "model_type"' in text:
        return False
    return True


def _extract_knobs(summary: str) -> dict[str, Any]:
    marker = "knobs="
    if marker not in summary:
        return {}
    raw = summary.split(marker, 1)[1].strip()
    try:
        import ast

        value = ast.literal_eval(raw.split("Dropped")[0].strip().rstrip("."))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def _tool_return_text(event: FunctionToolResultEvent) -> tuple[str, str]:
    """Tool output lives on result/part, not event.content (that's optional extra user text)."""
    result = getattr(event, "result", None)
    if result is None:
        result = getattr(event, "part", None)
    call_id = str(getattr(event, "tool_call_id", "") or "")
    if result is not None:
        call_id = call_id or str(getattr(result, "tool_call_id", "") or getattr(result, "id", "") or "")
        raw = getattr(result, "content", None)
        if isinstance(raw, str) and raw.strip():
            return call_id, raw.strip()
        getter = getattr(result, "model_response_str", None)
        if callable(getter):
            try:
                text = getter()
                if isinstance(text, str) and text.strip():
                    return call_id, text.strip()
            except Exception:
                pass
        if raw is not None:
            return call_id, str(raw).strip()
    return call_id, ""


def _map_agent_event(event: object) -> dict[str, Any] | None:
    try:
        if isinstance(event, PartStartEvent) and isinstance(event.part, TextPart):
            text = event.part.content or ""
            if text:
                return {"type": "text_delta", "delta": text}
            return None
        if isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
            delta = event.delta.content_delta or ""
            if delta:
                return {"type": "text_delta", "delta": delta}
            return None
        if isinstance(event, FunctionToolCallEvent):
            part = event.part
            name = getattr(part, "tool_name", "") if part else ""
            args = getattr(part, "args", {}) if part else {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {"raw": args}
            call_id = getattr(part, "tool_call_id", None) or getattr(part, "id", None) or uuid.uuid4().hex[:8]
            return {"type": "tool_start", "id": call_id, "name": name, "args": args or {}}
        if isinstance(event, FunctionToolResultEvent):
            call_id, content = _tool_return_text(event)
            ok = _tool_succeeded(content)
            return {"type": "tool_end", "id": call_id, "ok": ok, "summary": content[:2000]}
        if isinstance(event, ToolCallPart):
            return {
                "type": "tool_start",
                "id": event.tool_call_id or event.id or uuid.uuid4().hex[:8],
                "name": event.tool_name,
                "args": event.args if isinstance(event.args, dict) else {},
            }
        return None
    except Exception:
        logger.exception("failed to map agent event %s", type(event).__name__)
        return None
