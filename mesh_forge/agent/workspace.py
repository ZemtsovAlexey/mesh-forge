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
    return build_workspace_brief(
        ctx.deps.store,
        ctx.deps.chat_id,
        ctx.deps.attachments,
        reply_artifacts=ctx.deps.reply_artifacts,
    )


def collect_message_artifacts(message: UiMessage) -> list[Artifact]:
    items: list[Artifact] = []
    seen: set[str] = set()
    for art in [*message.attachments, *message.artifacts]:
        key = art.id or art.name
        if key and key not in seen:
            seen.add(key)
            items.append(art)
    for tool in message.tools:
        for art in tool.artifacts:
            key = art.id or art.name
            if key and key not in seen:
                seen.add(key)
                items.append(art)
    return items


def cited_artifacts(
    messages: list[UiMessage],
    reply_to: str,
    reply_artifact_ids: list[str] | None = None,
) -> list[Artifact]:
    mid = (reply_to or "").strip()
    if not mid:
        return []
    target = next((m for m in messages if m.id == mid), None)
    if target is None:
        return []
    items = collect_message_artifacts(target)
    wanted = [str(x).strip() for x in (reply_artifact_ids or []) if str(x).strip()]
    if not wanted:
        return items
    allow = set(wanted)
    return [a for a in items if a.id in allow or a.name in allow]


def format_reply_prompt(artifacts: list[Artifact], reply_to: str) -> str:
    if not artifacts:
        return f"User is replying to chat message {reply_to}. Use that result, not a new object."
    lines = [f"User is replying to chat message {reply_to}. Work on THESE artifacts only:"]
    images: list[str] = []
    front_id = ""
    for art in artifacts:
        view = (art.view or art.label or "").strip()
        extra = f" view={view}" if view else ""
        lines.append(f"- {art.kind} id={art.id}{extra}")
        if art.kind == "image":
            images.append(art.id)
            if view.lower() == "front" and not front_id:
                front_id = art.id
    if images:
        quoted = ", ".join(repr(i) for i in images)
        lines.append(
            f"look refs=[{quoted}]. If the photo has floor/studio/background, "
            f"remove_background images=[{quoted}] then images_to_mesh with the NEW cutout ids. "
            f"Else images_to_mesh images=[{quoted}]."
        )
    if front_id:
        lines.append(
            f"If the user wants more angles from this front, generate_views(ref_image={front_id!r}). "
            "If they want to redo the picture itself, generate_image with a new seed."
        )
    if any(a.kind == "mesh" for a in artifacts):
        lines.append(
            "If they comment on the mesh shape, look(target='mesh', views='orbit') "
            "or look with zoom/region for a detail. "
            "Extra wings/blobs/protrusions → carve_mesh, not generate_image. "
            "If a recent edit made it worse, restore_mesh. "
            "Do not generate_image unless they ask to redo the picture."
        )
    return "\n".join(lines)


def build_workspace_brief(
    store: ChatStore,
    chat_id: str,
    attachments: list[Artifact] | None = None,
    reply_artifacts: list[Artifact] | None = None,
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
    source = store.source_mesh(chat_id)
    previous = store.previous_mesh(chat_id)
    if mesh:
        lines.append(f"- Current mesh: {mesh.name}")
        if source:
            extra = " (same as current)" if source.name == mesh.name else ""
            lines.append(f"- Source mesh (Hunyuan/upload, restore_mesh to='source'): {source.name}{extra}")
        if previous and previous.name != mesh.name:
            lines.append(f"- Previous mesh (before last edit, restore_mesh to='previous'): {previous.name}")
        lines.append(
            "- Mesh-edit mode: do NOT generate_image / generate_views / images_to_mesh "
            "unless the user explicitly asks to redo the picture or rebuild the mesh. "
            "Extra volumes/wings → carve_mesh after look. "
            "If an edit ruined the shape, restore_mesh — do not repair the broken result and do not regen."
        )
    else:
        lines.append("- Current mesh: none")
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
        if any(a.kind == "image" for a in attached) and not mesh:
            lines.append(
                "  If those photos have floor/studio/background, remove_background first, "
                "then images_to_mesh with the NEW cutout ids."
            )
    cited = reply_artifacts or []
    if cited:
        bits = ", ".join(
            f"{a.kind}:{a.id}" + (f"({a.view or a.label})" if (a.view or a.label) else "")
            for a in cited
        )
        lines.append(f"- User reply target: {bits}")
        if mesh:
            lines.append(
                "  Work on these ids. Mesh comments → look / inspect_mesh / carve_mesh / restore_mesh. "
                "Do not images_to_mesh unless they ask to rebuild."
            )
        else:
            lines.append(
                "  Use these ids for look / remove_background / images_to_mesh / generate_views(ref_image=...)."
            )
    if len(lines) == 1:
        return ""
    if mesh:
        lines.append(
            "Short follow-ups continue this mesh. "
            "Do not generate_image / images_to_mesh unless the user asks to redo the picture. "
            "If an edit made the shape worse, restore_mesh(to='previous' or 'source'). "
            "Do not ask for a new object description."
        )
    else:
        lines.append(
            "Short follow-ups (продолжи / дальше / ok) continue this job. "
            "If look already returned NEXT: mesh, images_to_mesh with those ids. "
            "If NEXT: cutout, remove_background first. "
            "If NEXT: regen, generate_image with a new seed (also for 3/4 or warped geometry). "
            "Do not ask for a new object description."
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
