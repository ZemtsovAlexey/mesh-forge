from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Any

from openai import OpenAI

from mesh_forge.config import AppConfig, load_config

DEFAULT_CREATE_QUESTIONS = [
    "Что именно моделируем? (объект / персонаж / здание…)",
    "Какой стиль? (мультяшный / реалистичный / clay / lowpoly…)",
    "Какие детали обязательны?",
]

PLANNER_SYSTEM = """You are a 3D mesh operations planner for 3D printing.
Given a user instruction and optional mesh stats, output ONLY valid JSON:
{
  "operations": [
    {"op": "scale_axis", "axis": "z", "value_mm": 100},
    {"op": "scale_uniform", "factor": 1.1},
    {"op": "solidify", "thickness_mm": 3},
    {"op": "decimate", "target_faces": 200000},
    {"op": "smooth", "iterations": 2}
  ],
  "summary": "short explanation"
}
Allowed ops: scale_axis (x|y|z + value_mm), scale_uniform (factor), solidify (thickness_mm),
decimate (target_faces), smooth (iterations), fill_holes, remove_needles, remesh_voxel (voxel_mm).

Rules:
- Prefer minimal ops. For "remove spikes / needles / noise / jagged / fix edges" use
  remove_needles then smooth (1-3), optional decimate — do NOT remesh_voxel unless the user
  explicitly asks to remesh.
- remesh_voxel is destructive on large scans (bbox > 200 mm). If you must remesh, set
  voxel_mm to at least max(bbox)/150 (often 5–20 mm), never 0.5 mm on meter-scale parts.
- fill_holes when user asks to close holes (works best under ~200k faces; still include it).
- Never invent dimensions; use mesh stats bbox when scaling.
If you cannot map the request, return empty operations and explain in summary."""


class LMStudioClient:
    def __init__(self, config: AppConfig | None = None):
        self.config = config or load_config()
        self.client = OpenAI(
            base_url=self.config.llm.base_url,
            api_key=self.config.llm.api_key,
        )

    def chat(self, model: str, messages: list[dict[str, Any]], temperature: float = 0.2) -> str:
        from mesh_forge import progress as prog
        from mesh_forge.runtime import get_gpu_scheduler

        project_id = prog.current_project_id()
        with get_gpu_scheduler().acquire("LM Studio", kind="llm", project_id=project_id):
            prog.raise_if_cancelled(project_id)
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
            )
            return response.choices[0].message.content or ""

    def plan_edit(self, instruction: str, mesh_stats: dict[str, Any] | None = None) -> dict[str, Any]:
        user_content = f"Instruction: {instruction}\n"
        if mesh_stats:
            user_content += f"Mesh stats: {json.dumps(mesh_stats, ensure_ascii=False)}\n"
        text = self.chat(
            self.config.llm.planner_model,
            [
                {"role": "system", "content": PLANNER_SYSTEM},
                {"role": "user", "content": user_content},
            ],
        )
        return _parse_json_response(text)

    def clarify_or_enhance(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        system = """You help prepare a short English subject line for text-to-3D image generation.
Reply with ONLY valid JSON:
{
  "intent": "create",
  "assistant_message": "short reply in the user's language",
  "questions": ["up to 3 short clarifying questions"],
  "ready": false,
  "draft_prompt_en": "",
  "confidence": 0.0
}

Clarify rules (draft_prompt_en in your JSON is ignored — server translates user text):
- Prefer ready=true when the subject is clear. Ask questions ONLY if critical.
- If ready=false: questions MUST contain 1-3 real question strings; assistant_message lists them.
- If ready=true: questions=[], fill draft_prompt_en; assistant_message must be empty or a
  very short status in the user's language (e.g. «Ок, генерирую.») — NEVER paste
  draft_prompt_en or any English translation into assistant_message (translation is
  internal to the generator).
- Never map the request to mesh filters (smooth/decimate/solidify).
- If the user complains that you asked nothing / wants to proceed, set ready=true with a
  faithful short draft from the conversation so far."""
        text = self.chat(
            self.config.llm.planner_model,
            [{"role": "system", "content": system}, *messages],
            temperature=0.2,
        )
        data = _parse_json_response(text)
        if data.get("parse_error"):
            return {
                "intent": "create",
                "assistant_message": str(data.get("summary") or text)[:500],
                "questions": list(DEFAULT_CREATE_QUESTIONS),
                "ready": False,
                "draft_prompt_en": "",
                "confidence": 0.0,
            }
        data["intent"] = "create"
        questions = [str(q).strip() for q in (data.get("questions") or []) if str(q).strip()][:3]
        user_subject = _user_subject_from_messages(messages)
        ready = bool(data.get("ready"))
        if ready and user_subject:
            draft = self.ensure_english_subject(user_subject)
            ready = bool(draft)
        else:
            draft = ""
            ready = False
        assistant = str(data.get("assistant_message") or "").strip()

        # Empty "I'll ask questions" responses are useless — force defaults or draft.
        if not ready and not questions:
            questions = list(DEFAULT_CREATE_QUESTIONS)
        if ready:
            questions = []
        data["questions"] = questions
        data["draft_prompt_en"] = draft
        data["ready"] = ready
        data["assistant_message"] = assistant
        return data

    def ensure_english_subject(self, text: str) -> str:
        """Literal English translation for ComfyUI — trim only, no embellishment."""
        cleaned = _trim_subject_prompt(text)
        if not cleaned:
            return ""
        if _looks_english(cleaned):
            return cleaned
        system = (
            "You are a literal translator.\n"
            "Translate the USER message into English for image generation.\n"
            "Reply with ONLY the translation — no quotes, no JSON, no explanation.\n"
            "CRITICAL: translate faithfully. Do not add, remove, or change any detail."
        )
        try:
            out = self.chat(
                self.config.llm.planner_model,
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": cleaned},
                ],
                temperature=0.0,
            ).strip().strip('"').strip("'")
            if out.lower().startswith("draft:"):
                out = out.split(":", 1)[1].strip()
            if out.startswith("{") and "draft" in out.lower():
                parsed = _parse_json_response(out)
                out = str(parsed.get("draft_prompt_en") or parsed.get("text") or out).strip()
            out = _trim_subject_prompt(out)
            if out and _looks_english(out):
                return out
        except Exception:
            pass
        return ""

    def interpret_mesh_edit(
        self,
        *,
        instruction: str,
        mesh_preview_b64: str,
        prior_prompt: str = "",
        prefer_guided: bool = True,
        reference_photo_b64: str | None = None,
    ) -> dict[str, Any]:
        guided_bias = (
            "Default to guided_edit for add/remove detail, missing parts, small shape fixes."
            if prefer_guided
            else "Prefer semantic_edit only when the user wants a major redesign."
        )
        system = f"""You analyze a 3D mesh preview (and optional reference photo) plus a user edit request.
Classify the request:
- geometry_edit: cleanup of the EXISTING mesh (noise, spikes, holes, smooth, repair, jagged edges).
- guided_edit: keep the same object identity, apply a local change (add door/window/ear, remove a bump).
- semantic_edit: major redesign / new pose / start over (full text-to-3D regen, identity not preserved).

{guided_bias}

Reply with ONLY valid JSON:
{{
  "intent": "geometry_edit" | "guided_edit" | "semantic_edit",
  "assistant_message": "short reply in the user's language",
  "questions": [],
  "ready": false,
  "edit_brief_en": "",
  "confidence": 0.0
}}

Rules:
- geometry_edit: ready=true; edit_brief_en = short English cleanup summary.
- guided_edit: ready=true; edit_brief_en = one English sentence describing the object AFTER the change.
  CRITICAL: identify the real subject from the reference PHOTO and/or prior prompt — NOT from the grey
  mesh preview alone (previews look like generic clay blobs).
  Keep colors/style/identity. Mention the fix explicitly. Do not invent a different subject.
- semantic_edit: edit_brief_en describes the new object for full regeneration; include the fix explicitly.
- Prefer guided_edit when both a detail change and vague "make nicer" appear together.
- If unclear, ready=false and ask up to 3 questions."""
        user_text = f"User edit request: {instruction.strip()}"
        if prior_prompt.strip():
            user_text += f"\nPrior creation prompt/context: {prior_prompt.strip()}"
        if reference_photo_b64:
            user_text += (
                "\nA reference photo is attached — use it as ground truth for what the object should look like."
            )
        content: list[dict[str, Any]] = [
            {"type": "text", "text": user_text},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{mesh_preview_b64}"}},
        ]
        if reference_photo_b64:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{reference_photo_b64}"},
            })
        text = self.chat(
            self.config.llm.vision_model,
            [
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
            temperature=0.2,
        )
        data = _parse_json_response(text)
        if data.get("parse_error"):
            return {
                "intent": "guided_edit" if prefer_guided else "semantic_edit",
                "assistant_message": str(data.get("summary") or text)[:500],
                "questions": ["Что именно нужно изменить в модели?"],
                "ready": False,
                "edit_brief_en": "",
                "confidence": 0.0,
            }
        intent = str(data.get("intent") or ("guided_edit" if prefer_guided else "semantic_edit")).strip().lower()
        if intent not in {"geometry_edit", "guided_edit", "semantic_edit"}:
            intent = "guided_edit" if prefer_guided else "semantic_edit"
        data["intent"] = intent
        data["questions"] = list(data.get("questions") or [])[:3]
        data["edit_brief_en"] = str(data.get("edit_brief_en") or "").strip()
        if intent in {"geometry_edit", "guided_edit"}:
            data["ready"] = True if data["edit_brief_en"] or intent == "geometry_edit" else bool(data.get("ready"))
            if intent == "guided_edit" and not data["edit_brief_en"]:
                data["ready"] = False
        else:
            data["ready"] = bool(data.get("ready")) and bool(data["edit_brief_en"])
        data["assistant_message"] = str(data.get("assistant_message") or "").strip()
        return data

    def generate_openscad(self, prompt: str) -> str:
        system = (
            "Write OpenSCAD code for a 3D-printable object. "
            "Use only basic primitives and difference/union. "
            "Output ONLY OpenSCAD code, no markdown."
        )
        return self.chat(
            self.config.llm.planner_model,
            [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        ).strip()

    def inspect_images(
        self,
        images: list[tuple[str, Path]],
        *,
        question: str = "",
    ) -> str:
        """On-demand VLM look for the chat agent (Russian)."""
        parts: list[dict[str, Any]] = []
        for label, path in images:
            encoded = _encode_image(path)
            if not encoded:
                continue
            mime, b64 = encoded
            parts.append({"type": "text", "text": f"[image {label}]"})
            parts.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})
        if not parts:
            return ""
        prompt = (
            "Ты глаза 3D-агента. Опиши то, что видишь, по-русски: объект, ракурс/поза, "
            "стиль, пропорции, заметные дефекты, лишнее, соответствие запросу. "
            "Без вступлений. Если кадров несколько — сначала общее, потом отличия по меткам."
        )
        focus = (question or "").strip()
        if focus:
            prompt += f"\nВопрос агента: {focus}"
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}, *parts]
        try:
            return self.chat(
                self.config.llm.vision_model,
                [{"role": "user", "content": content}],
                temperature=0.2,
            ).strip()
        except Exception:
            return ""

    def describe_image(
        self,
        image_path: Path | None = None,
        *,
        image_b64: str | None = None,
        context: str = "",
        mime: str = "image/png",
    ) -> str:
        """Short vision caption for a single image (Russian)."""
        if image_path and image_path.is_file() and not image_b64:
            return self.inspect_images([(context or image_path.name, image_path)], question="")
        if not image_b64:
            return ""
        tmp_label = context or "image"
        hint = (context or "").strip()
        prompt = (
            "Кратко опиши изображение для 3D-агента (1–2 предложения по-русски): "
            "что за объект, поза/ракурс, стиль, цвета, заметные дефекты или лишнее. "
            "Без вступлений и списков."
        )
        if hint:
            prompt += f"\nКонтекст: {hint}"
        content: list[dict[str, Any]] = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_b64}"}},
        ]
        try:
            return self.chat(
                self.config.llm.vision_model,
                [{"role": "user", "content": content}],
                temperature=0.2,
            ).strip()
        except Exception:
            return ""

    def describe_image_diff(self, instruction: str, mesh_preview_b64: str | None, ref_image_path: Path | None) -> str:
        content: list[dict[str, Any]] = [{"type": "text", "text": instruction}]
        if mesh_preview_b64:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{mesh_preview_b64}"},
            })
        if ref_image_path and ref_image_path.is_file():
            b64 = base64.b64encode(ref_image_path.read_bytes()).decode()
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })
        return self.chat(
            self.config.llm.vision_model,
            [{"role": "user", "content": content}],
        )

    def health_check(self) -> bool:
        try:
            self.client.models.list()
            return True
        except Exception:
            return False

    def list_models(self) -> list[str]:
        """Return model IDs exposed by LM Studio (loaded models)."""
        try:
            response = self.client.models.list()
            ids = [m.id for m in response.data if getattr(m, "id", None)]
            return sorted(set(ids))
        except Exception:
            return []

    def models_status(self) -> str:
        if not self.health_check():
            return "LM Studio API недоступен. Запустите Local Server в LM Studio."
        models = self.list_models()
        if not models:
            return (
                "API отвечает, но моделей нет. Загрузите модель в LM Studio "
                "(Chat → Load model) и обновите список."
            )
        lines = [f"Доступно моделей: {len(models)}", ""]
        for mid in models:
            mark = []
            if mid == self.config.llm.planner_model:
                mark.append("planner")
            if mid == self.config.llm.vision_model:
                mark.append("vision")
            suffix = f" ← {', '.join(mark)}" if mark else ""
            lines.append(f"  • {mid}{suffix}")
        return "\n".join(lines)


def _encode_image(path: Path) -> tuple[str, str] | None:
    if not path or not path.is_file():
        return None
    suffix = path.suffix.lower()
    mime = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".png": "image/png",
    }.get(suffix, "image/png")
    return mime, base64.b64encode(path.read_bytes()).decode()


def _parse_json_response(text: str) -> dict[str, Any]:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"operations": [], "summary": text, "parse_error": True}


_VIEW_PREFIXES = ("front:", "left:", "back:", "right:", "view:")


def _trim_subject_prompt(text: str) -> str:
    """Whitespace and accidental view-prefix cleanup only — no semantic edits."""
    cleaned = (text or "").strip().strip('"').strip("'")
    if not cleaned:
        return ""
    for prefix in _VIEW_PREFIXES:
        if cleaned.lower().startswith(prefix):
            cleaned = cleaned[len(prefix) :].strip()
    return re.sub(r"\s{2,}", " ", cleaned).strip()


def _user_subject_from_messages(messages: list[dict[str, Any]]) -> str:
    bits = [str(m.get("content") or "").strip() for m in messages if m.get("role") == "user"]
    return "\n".join(b for b in bits if b).strip()


def _looks_english(text: str) -> bool:
    letters = [c for c in (text or "") if c.isalpha()]
    if not letters:
        return False
    ascii_letters = sum(1 for c in letters if ord(c) < 128)
    return ascii_letters / len(letters) > 0.85
