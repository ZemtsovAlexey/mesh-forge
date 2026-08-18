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
_REJECTION = re.compile(
    r"^(нет+|неа|не надо|не нужно|не то|неправильно|мимо|no+|nope|wrong|not this)\b[\s!.…]*.*$",
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


def is_rejection(text: str) -> bool:
    value = (text or "").strip()
    if not value:
        return False
    return bool(_REJECTION.match(value))


def workspace_instructions(ctx: RunContext[ChatDeps]) -> str:
    return build_workspace_brief(
        ctx.deps.store,
        ctx.deps.chat_id,
        ctx.deps.attachments,
        reply_artifacts=ctx.deps.reply_artifacts,
        looks_without_edit=int(getattr(ctx.deps, "looks_without_edit", 0) or 0),
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
            "If they comment on the mesh shape, look(target='mesh') "
            "then the matching edit tool. "
            "If a recent edit made it worse, restore_mesh. "
            "generate_image if they want to redo the picture."
        )
    return "\n".join(lines)


def build_workspace_brief(
    store: ChatStore,
    chat_id: str,
    attachments: list[Artifact] | None = None,
    reply_artifacts: list[Artifact] | None = None,
    looks_without_edit: int = 0,
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
        hist = store.mesh_history(chat_id)
        if len(hist) > 1:
            lines.append(
                f"- Undo stack: {len(hist)} older mesh(es). "
                "restore_mesh(to='previous') walks back one each call."
            )
        region, pick = store.active_mesh_target(chat_id)
        if pick and len(pick) >= 3:
            nx, ny, nz = pick[0], pick[1], pick[2]
            topo = store.active_mesh_topo(chat_id)
            from mesh_forge.ops.topo import format_topo, topo_valid

            extra = ""
            if topo_valid(topo):
                extra = (
                    f" Topology: {format_topo(topo)} on mesh "
                    f"{topo.get('mesh') or 'current'}. "
                    "Paint with mask_mesh (omit x,y — uses this click). "
                    "Show the red overlay and STOP. remove_mesh only after the user confirms."
                )
            lines.append(
                f"- Current region: user click at "
                f"normalized {nx:.2f},{ny:.2f},{nz:.2f}.{extra} "
                "look(target='mesh') without region, then the edit tool without region."
            )
        elif region:
            lines.append(
                f"- Current region: {region}. "
                "Omit region on look. For deletion paint mask_mesh then remove_mesh."
            )
        meta = store.get_meta(chat_id)
        mask_info = dict(meta.mesh_mask or {})
        mask_state = dict(meta.mask_state or {})
        removal_state = dict(meta.removal_state or {})
        if removal_state:
            strategy = str(removal_state.get("strategy") or "")
            status = str(removal_state.get("proposal_status") or "ready")
            mesh_name = str(removal_state.get("mesh") or mesh.name)
            lines.append(
                f"- Removal proposal: strategy={strategy or 'unknown'} on {mesh_name} (proposal={status}). "
                "This is the new universal delete workflow. Review the proposal preview with the user. "
                "After confirmation call remove_extra(apply=True). "
                "If the proposal looks wrong, you may rebuild with remove_extra(describe=...) or use mask_mesh only for a surface patch."
            )
            latest_user = _latest_user_text(messages)
            if latest_user and is_rejection(latest_user):
                lines.append(
                    "- The latest user message rejects the current removal proposal. Do NOT call remove_extra(apply=True). "
                    "Clear/rebuild the proposal instead."
                )
        painted = int(mask_info.get("count") or 0)
        if painted > 0 and painted < 8:
            lines.append(
                f"- Painted mask: {painted} faces on "
                f"{mask_info.get('mesh') or 'current'} — TOO SMALL, automatic proposal is not reliable. "
                "Do NOT call remove_mesh. Ask the user to click the extra bit on a look PNG, then mask_mesh again."
            )
        elif painted > 8000:
            lines.append(
                f"- Painted mask: {painted} faces on "
                f"{mask_info.get('mesh') or 'current'} — TOO LARGE, likely the skirt/body. "
                "Do NOT call remove_mesh. Ask the user to click the petal on a look PNG, then mask_mesh again."
            )
        elif painted > 0:
            status = str(mask_state.get("proposal_status") or "ready")
            verdict = str(mask_state.get("review_verdict") or "")
            verdict_txt = f" review={verdict}." if verdict else ""
            lines.append(
                f"- Painted mask: {painted} faces on "
                f"{mask_info.get('mesh') or 'current'} (red overlay, proposal={status}).{verdict_txt} "
                "mask_mesh already did multi-view detection, geometry checks, and strict auto-review. "
                "Wait for user confirmation before remove_mesh. If the proposal is tiny, partial, or doubtful, ask for a click and rerun mask_mesh."
            )
        look = dict(meta.look_view or {})
        if look.get("aim_x") is not None and look.get("aim_y") is not None:
            lines.append(
                f"- Viewport aim: {look.get('views') or 'view'} "
                f"({float(look['aim_x']):.2f},{float(look['aim_y']):.2f}). "
                "This click can be used as fallback if automatic masking is unsure."
            )
        lines.append(
            "- Mesh-edit mode: remove_extra is now the primary delete entrypoint. It chooses a strategy (protrusion_cut / island_drop / edge_trim / surface_patch), builds a proposal, and waits for confirmation before applying. "
            "mask_mesh remains as a specialized surface-patch fallback and legacy workflow. "
            "remove_extra(apply=True) is preferred after confirmation; remove_mesh is kept for already-painted red masks and backward compatibility."
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
                "  Work on these ids. Mesh comments → look / mask_mesh / remove_mesh / restore_mesh. "
                "images_to_mesh if they ask to rebuild."
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
            "Delete a part → remove_extra, review the proposal with the user, then remove_extra(apply=True) if they confirm. "
            "Use mask_mesh mainly for surface patches or legacy painted-mask flows. "
            "If identity is lost, restore_mesh(to='previous' or 'source'). "
            "generate_image / images_to_mesh only if they ask to start over."
        )
    else:
        lines.append(
            "Short follow-ups (продолжи / дальше / ok) continue this job. "
            "If look already returned NEXT: mesh, images_to_mesh with those ids. "
            "If NEXT: cutout, remove_background first. "
            "If NEXT: regen, generate_image with a new seed (also for 3/4 or warped geometry)."
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
        if text and not is_followup(text) and not is_rejection(text):
            return text.splitlines()[0][:240]
    for message in reversed(messages):
        if message.role != "assistant":
            continue
        for tool in reversed(message.tools):
            prompt = tool.args.get("prompt") if isinstance(tool.args, dict) else None
            if isinstance(prompt, str) and prompt.strip():
                return prompt.strip()[:240]
    return ""


def _latest_user_text(messages: list[UiMessage]) -> str:
    for message in reversed(messages):
        if message.role == "user":
            return (message.content or "").strip()
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
