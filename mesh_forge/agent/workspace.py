from __future__ import annotations

import re
from dataclasses import replace
from typing import Any

from pydantic_ai.messages import ModelRequest, SystemPromptPart, ToolReturnPart, UserPromptPart
from pydantic_ai.tools import RunContext

from mesh_forge import progress as prog
from mesh_forge.agent.deps import ChatDeps
from mesh_forge.chat.models import Artifact, ToolCallRecord, UiMessage
from mesh_forge.chat.store import ChatStore

_FOLLOWUP = re.compile(
    r"^(продолж\w*|дальше|далее|ok|okay|ок|угу|ага|да+|yes|go|next|continue|"
    r"ещё|еще|повтор\w*|redo|retry|переделай|заново|го|жди)[\s!.…]*$",
    re.IGNORECASE,
)
_MAX_TOOL_RETURN = 480
_KEEP_FULL_RETURNS = 4
_USER_MARK = "Latest user message: "


def is_followup(text: str) -> bool:
    value = (text or "").strip()
    if not value:
        return True
    if len(value) > 48:
        return False
    return bool(_FOLLOWUP.match(value))


def workspace_instructions(ctx: RunContext[ChatDeps]) -> str:
    return build_workspace_brief(ctx.deps.store, ctx.deps.chat_id, ctx.deps.attachments)


def build_workspace_brief(
    store: ChatStore,
    chat_id: str,
    attachments: list[Artifact] | None = None,
) -> str:
    """Compact chat state for the local LLM — it often ignores long tool history."""
    lines = ["Current workspace (trust this; do not re-ask what to create if a goal or images exist):"]
    messages = _safe_messages(store, chat_id)
    goal = _goal_from_messages(messages)
    if goal:
        lines.append(f"- Goal: {goal}")
    files = _file_lines(store, chat_id)
    if files:
        lines.append("- Files:")
        lines.extend(files)
    mesh = store.current_mesh(chat_id)
    lines.append(f"- Current mesh: {mesh.name}" if mesh else "- Current mesh: none")
    tools = _recent_tools(messages)
    if tools:
        lines.append("- Recent tools:")
        lines.extend(tools)
    state = prog.get(chat_id)
    if state and state.active:
        lines.append(f"- In progress: {state.operation} {state.stage}".strip())
    attached = attachments or []
    if attached:
        bits = ", ".join(f"{a.kind}:{a.id}" for a in attached)
        lines.append(f"- This-turn attachments: {bits}")
    if len(lines) == 1:
        return ""
    lines.append(
        "Short follow-ups (продолжи / дальше / ok) continue this job: "
        "use the image ids above for images_to_mesh, do not ask for a new object description."
    )
    return "\n".join(lines)


def with_workspace(prompt: str, brief: str) -> str:
    text = (prompt or "").strip() or "(see attached files)"
    if not brief:
        return text
    return f"{brief}\n\nLatest user message: {text}"


def compact_history(messages: list[Any]) -> list[Any]:
    """Trim old tool returns so the original user goal stays in the context window."""
    returns: list[ToolReturnPart] = []
    prompts: list[UserPromptPart] = []
    for message in messages:
        if not isinstance(message, ModelRequest):
            continue
        for part in message.parts:
            if isinstance(part, ToolReturnPart):
                returns.append(part)
            elif isinstance(part, UserPromptPart):
                prompts.append(part)
    keep_returns = {id(part) for part in returns[-_KEEP_FULL_RETURNS:]}
    keep_prompt = id(prompts[-1]) if prompts else None
    out: list[Any] = []
    for message in messages:
        if not isinstance(message, ModelRequest):
            out.append(message)
            continue
        new_parts = []
        changed = False
        for part in message.parts:
            if isinstance(part, SystemPromptPart):
                changed = True
                continue
            if isinstance(part, ToolReturnPart) and id(part) not in keep_returns:
                trimmed = _trim_tool_content(part.content)
                if trimmed is not None:
                    new_parts.append(replace(part, content=trimmed))
                    changed = True
                    continue
            if isinstance(part, UserPromptPart) and id(part) != keep_prompt:
                raw = _unwrap_user_prompt(part.content)
                if raw is not None:
                    new_parts.append(replace(part, content=raw))
                    changed = True
                    continue
            new_parts.append(part)
        if not new_parts:
            continue
        out.append(replace(message, parts=list(new_parts)) if changed else message)
    return out


def _safe_messages(store: ChatStore, chat_id: str) -> list[UiMessage]:
    try:
        return store.load_messages(chat_id)
    except Exception:
        return []


def _goal_from_messages(messages: list[UiMessage]) -> str:
    for message in reversed(messages):
        if message.role != "user":
            continue
        text = (message.content or "").strip()
        if text and not is_followup(text):
            return text.splitlines()[0][:240]
    for message in reversed(messages):
        if message.role != "assistant":
            continue
        for tool in reversed(message.tools):
            prompt = tool.args.get("prompt") if isinstance(tool.args, dict) else None
            if isinstance(prompt, str) and prompt.strip():
                return prompt.strip()[:240]
    return ""


def _file_lines(store: ChatStore, chat_id: str) -> list[str]:
    try:
        arts = store.list_files(chat_id)
    except Exception:
        return []
    lines: list[str] = []
    images = [a for a in arts if a.kind == "image"][-8:]
    meshes = [a for a in arts if a.kind == "mesh"][-4:]
    for art in images:
        extra = f" view={art.view}" if art.view else ""
        label = f" {art.label}" if art.label and art.label != art.id else ""
        lines.append(f"  - image {art.id}{extra}{label}")
    for art in meshes:
        lines.append(f"  - mesh {art.id}")
    return lines


def _recent_tools(messages: list[UiMessage]) -> list[str]:
    tools: list[ToolCallRecord] = []
    for message in reversed(messages):
        if message.role == "assistant" and message.tools:
            tools = list(message.tools)
            break
    lines: list[str] = []
    for tool in tools[-6:]:
        prompt = ""
        if isinstance(tool.args, dict):
            raw = tool.args.get("prompt")
            if isinstance(raw, str) and raw.strip():
                prompt = f" prompt={raw.strip()[:120]}"
            images = tool.args.get("images")
            if isinstance(images, list) and images:
                bits: list[str] = []
                for item in images:
                    if isinstance(item, dict):
                        ref = str(item.get("ref") or "")
                        view = str(item.get("view") or "")
                        bits.append(f"{view}:{ref}" if view else ref)
                    else:
                        bits.append(str(item))
                prompt += f" images={','.join(b for b in bits if b)}"
            elif images:
                prompt += f" images={images}"
        summary = (tool.summary or "").strip().replace("\n", " ")
        if summary:
            summary = f" -> {summary[:160]}"
        lines.append(f"  - {tool.name} {tool.status}{prompt}{summary}".rstrip())
    return lines


def _trim_tool_content(content: Any) -> str | None:
    if not isinstance(content, str) or len(content) <= _MAX_TOOL_RETURN:
        return None
    return content[:_MAX_TOOL_RETURN].rstrip() + "…"


def _unwrap_user_prompt(content: Any) -> str | None:
    if not isinstance(content, str) or _USER_MARK not in content:
        return None
    return content.rsplit(_USER_MARK, 1)[-1].strip() or None
