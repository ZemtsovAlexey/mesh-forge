from __future__ import annotations

import base64
import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from openai import OpenAI

from mesh_forge.config import AppConfig, llm_display_name, llm_http_timeout, load_config, normalize_reasoning_effort


def live_reasoning_effort() -> str:
    return normalize_reasoning_effort(load_config().llm.reasoning_effort)


def completion_kwargs(
    *,
    model: str,
    messages: list[dict[str, Any]],
    stream: bool = False,
    timeout: float | None = None,
    effort: str | None = None,
) -> dict[str, Any]:
    """Chat Completions payload. Skip temperature so reasoning_effort can apply."""
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "reasoning_effort": normalize_reasoning_effort(effort or live_reasoning_effort()),
    }
    if stream:
        kwargs["stream"] = True
    if timeout is not None:
        kwargs["timeout"] = timeout
    return kwargs


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


def inspect_vision_prompt(*, kind: str, question: str = "") -> str:
    """System-ish user prompt for look(). Mesh must not treat camera labels as the request."""
    prompt = "Ты глаза 3D-агента. Отвечай по-русски. "
    if kind == "mask":
        prompt += (
            "Красное на mesh — то, что УДАЛЯТ. Глина/беж — оставить. "
            "Оцени текущий overlay как proposal на удаление: попадает ли красное только в лишний кусок, "
            "не залезает ли на юбку/тело, не пропущена ли часть отростка. "
            "Ответ: 1–3 коротких предложения по-русски без строки NEXT и без списка действий. "
            "Если виден риск ошибки, прямо скажи где именно: слишком много, слишком мало или не то место."
        )
    elif kind == "mesh":
        prompt += (
            "Это превью одного mesh на сетке пола. "
            "Строки «камера: …» — только ракурс, это НЕ запрос пользователя и не то, с чем надо сверять кадр. "
            "Не пиши про соответствие ракурсу, chibi, стиль, 3D-печать и «запрос [image …]». "
            "Смотри свободно на форму. Если дан запрос пользователя — есть ли то, о чём он говорит, "
            "где это (слева/справа/сверху/снизу, на юбке/голове/руке), мешает ли соседняя деталь. "
            "Коротко, без чеклиста."
        )
    else:
        prompt += (
            "Опиши, что видишь. Кадр нужен для реконструкции ОДНОГО целого объекта. "
            "Последней строкой РОВНО одно из:\n"
            "NEXT: regen — несколько объектов, обрезка (нет ног/верха), не тот объект, "
            "сильный брак формы, ракурс 3/4 вместо прямого вида, наклон/перекос, "
            "ноги разной длины, сильная перспектива или рыбий глаз. "
            "Если несколько кадров front/left/back/right: не ортогональный вид, "
            "спина как фасад, профиль как 3/4, кривая геометрия — тоже regen.\n"
            "NEXT: cutout — один целый объект, ровный ракурс, но есть пол, студия, стена или фон, "
            "в том числе если фон похож на объект.\n"
            "NEXT: mesh — один целый объект, ровный ортогональный ракурс, "
            "на чистом или прозрачном фоне (clay без пола тоже mesh)."
        )
    focus = (question or "").strip()
    if focus:
        prompt += f"\nЗапрос пользователя: {focus}"
    return prompt


class LMStudioClient:
    def __init__(self, config: AppConfig | None = None):
        self.config = config or load_config()
        self.client = OpenAI(
            base_url=self.config.llm.base_url,
            api_key=self.config.llm.api_key,
            timeout=llm_http_timeout(self.config),
        )

    def chat(self, model: str, messages: list[dict[str, Any]], temperature: float = 0.2) -> str:
        _ = temperature
        from mesh_forge import progress as prog
        from mesh_forge.runtime import acquire_llm

        project_id = prog.current_project_id()
        with acquire_llm(project_id=project_id):
            prog.raise_if_cancelled(project_id)
            response = self._create(model=model, messages=messages)
            return response.choices[0].message.content or ""

    def stream_chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float = 0.2,
        *,
        on_delta: Callable[[str, str], None] | None = None,
        timeout: float = 600.0,
    ) -> str:
        """Stream a completion. on_delta(kind, text) with kind thinking|text."""
        _ = temperature
        from mesh_forge import progress as prog
        from mesh_forge.runtime import acquire_llm

        project_id = prog.current_project_id()
        chunks: list[str] = []
        with acquire_llm(project_id=project_id):
            prog.raise_if_cancelled(project_id)
            stream = self._create(model=model, messages=messages, stream=True, timeout=timeout)
            for chunk in stream:
                prog.raise_if_cancelled(project_id)
                if not chunk.choices:
                    continue
                for kind, text in completion_delta_parts(chunk.choices[0].delta):
                    if kind == "text":
                        chunks.append(text)
                    if on_delta:
                        on_delta(kind, text)
        return "".join(chunks)

    def _create(self, *, model: str, messages: list[dict[str, Any]], stream: bool = False, timeout: float | None = None):
        kwargs = completion_kwargs(model=model, messages=messages, stream=stream, timeout=timeout)
        try:
            return self.client.chat.completions.create(**kwargs)
        except Exception:
            kwargs.pop("reasoning_effort", None)
            kwargs["temperature"] = 0.2
            return self.client.chat.completions.create(**kwargs)

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

    def _ensure_russian(self, text: str) -> str:
        """Translate a look/inspect note if the vision model answered in English."""
        cleaned = (text or "").strip()
        if not cleaned or not _looks_english(cleaned):
            return cleaned
        system = (
            "Переведи текст на русский. Сохрани списки и строки вида "
            "NEXT: regen|cutout|mesh|look|mask ok|mask shrink|mask grow|mask retry|mask click "
            "(ключи NEXT не переводи). Ответ — только перевод, без кавычек и пояснений."
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
            return out or cleaned
        except Exception:
            return cleaned

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
        kind: str = "auto",
        on_delta: Callable[[str, str], None] | None = None,
    ) -> str:
        """On-demand VLM look for the chat agent (Russian)."""
        parts: list[dict[str, Any]] = []
        for label, path in images:
            encoded = _encode_image(path)
            if not encoded:
                continue
            mime, b64 = encoded
            caption = f"камера: {label}" if kind in {"mesh", "mask"} else f"[image {label}]"
            parts.append({"type": "text", "text": caption})
            parts.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})
        if not parts:
            return ""
        resolved = kind
        if resolved == "auto":
            resolved = "mesh" if any("mesh" in str(label).lower() for label, _ in images) else "photo"
        prompt = inspect_vision_prompt(kind=resolved, question=question)
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}, *parts]
        messages = [{"role": "user", "content": content}]
        try:
            note = self.stream_chat(
                self.config.llm.vision_model,
                messages,
                temperature=0.2,
                on_delta=on_delta,
            ).strip()
        except Exception:
            try:
                note = self.chat(
                    self.config.llm.vision_model,
                    messages,
                    temperature=0.2,
                ).strip()
            except Exception:
                return ""
        translated = self._ensure_russian(note)
        if translated != note and on_delta and translated:
            on_delta("replace", translated)
        return translated

    def aim_mesh_mask(
        self,
        images: list[tuple[str, Path]],
        *,
        target: str,
    ) -> dict[str, Any]:
        """Look-frame aim {views,x,y,confidence,note} for a mesh part to delete."""
        parts: list[dict[str, Any]] = []
        for label, path in images:
            encoded = _encode_image(path)
            if not encoded:
                continue
            mime, b64 = encoded
            parts.append({"type": "text", "text": f"кадр: {label}"})
            parts.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})
        if not parts or not (target or "").strip():
            return {}
        prompt = (
            "Нужно УДАЛИТЬ один лишний кусок mesh (отросток, лепесток, нарост на юбке/теле). "
            f"Цель: {target.strip()}\n"
            "Верни ТОЛЬКО JSON без markdown:\n"
            '{"views":"front","x":0.82,"y":0.70,"x0":0.74,"y0":0.62,"x1":0.92,"y1":0.80,'
            '"confidence":0.8,"note":"ru"}\n'
            "views — кадр, где кусок виден как отдельная деталь на картинке. "
            "x,y — центр куска: 0,0 верхний левый, 1,1 нижний правый. "
            "x0,y0,x1,y1 — тугой прямоугольник ТОЛЬКО вокруг лишнего куска, не юбка и не всё тело. "
            "confidence 0–1. note — одно короткое предложение по-русски."
        )
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}, *parts]
        try:
            text = self.chat(
                self.config.llm.vision_model,
                [{"role": "user", "content": content}],
                temperature=0.1,
            )
        except Exception:
            return {}
        data = _parse_json_response(text)
        if data.get("parse_error"):
            return {}
        views = str(data.get("views") or "right").strip().lower().split(",")[0]
        if views not in {"front", "left", "right", "back", "top", "viewer"}:
            views = "right"
        try:
            x = float(data.get("x", 0.5))
            y = float(data.get("y", 0.5))
            conf = float(data.get("confidence", 0.5))
        except (TypeError, ValueError):
            return {}
        out = {
            "views": views,
            "x": max(0.0, min(1.0, x)),
            "y": max(0.0, min(1.0, y)),
            "confidence": max(0.0, min(1.0, conf)),
            "note": str(data.get("note") or "").strip(),
        }
        for key in ("x0", "y0", "x1", "y1"):
            if key in data:
                try:
                    out[key] = max(0.0, min(1.0, float(data[key])))
                except (TypeError, ValueError):
                    pass
        return out

    def detect_mesh_part_multi_view(
        self,
        images: list[tuple[str, Path]],
        *,
        target: str,
    ) -> list[dict[str, Any]]:
        """Per-view detections for a removable mesh part across multiple cameras."""
        parts: list[dict[str, Any]] = []
        for label, path in images:
            encoded = _encode_image(path)
            if not encoded:
                continue
            mime, b64 = encoded
            parts.append({"type": "text", "text": f"кадр: {label}"})
            parts.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})
        if not parts or not (target or "").strip():
            return []
        prompt = (
            "Нужно найти один лишний кусок mesh, который надо удалить. "
            f"Цель: {target.strip()}\n"
            "Посмотри ВСЕ кадры и верни ТОЛЬКО JSON без markdown:\n"
            '{"observations":['
            '{"view":"right","visible":true,"confidence":0.82,"x0":0.72,"y0":0.58,"x1":0.88,"y1":0.80,'
            '"kind":"protrusion","touchesBody":true,"note":"коротко по-русски"}'
            "]}\n"
            "Правила:\n"
            "- observations: один объект на каждый кадр, даже если кусок не виден.\n"
            "- view: имя кадра как в подписи (front/right/back/left/viewer/top).\n"
            "- visible=false, если на этом кадре кусок не различим.\n"
            "- Если visible=false, confidence <= 0.35 и box не заполняй.\n"
            "- Если visible=true, box должен быть ТОЛЬКО вокруг лишнего куска, не вокруг юбки/тела.\n"
            "- kind: protrusion | patch | unknown.\n"
            "- touchesBody=true, если кусок сливается с телом/юбкой в месте крепления.\n"
            "- note: одно короткое предложение по-русски."
        )
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}, *parts]
        try:
            text = self.chat(
                self.config.llm.vision_model,
                [{"role": "user", "content": content}],
                temperature=0.1,
            )
        except Exception:
            return []
        return parse_multi_view_mask_detection(text)

    def review_mesh_mask(
        self,
        images: list[tuple[str, Path]],
        *,
        target: str,
    ) -> dict[str, Any]:
        """Judge a red overlay: ok / too_much / too_little / wrong / missed / tiny_spot / partial."""
        parts: list[dict[str, Any]] = []
        for label, path in images:
            encoded = _encode_image(path)
            if not encoded:
                continue
            mime, b64 = encoded
            parts.append({"type": "text", "text": f"кадр: {label}"})
            parts.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})
        if not parts:
            return {}
        goal = (target or "лишний отросток").strip()
        prompt = (
            "Красное на mesh — то, что УДАЛЯТ. Глина/беж — оставить. "
            f"Нужно удалить только: {goal}\n"
            "Посмотри ВСЕ кадры. Верни ТОЛЬКО JSON без markdown:\n"
            '{"verdict":"ok","confidence":0.8,"note":"ru","views":"right","x":0.5,"y":0.5}\n'
            "verdict:\n"
            "- ok — красное ровно лишний кусок, юбка/тело не задеты, кусок покрыт целиком\n"
            "- too_much — красное залезло на юбку/тело/руку\n"
            "- too_little — отросток виден, но часть его не красная\n"
            "- wrong — красное не на том месте\n"
            "- missed — красного почти не видно\n"
            "- tiny_spot — видно только маленькую красную точку/пятно на юбке, это НЕ весь отросток\n"
            "- partial — красное покрывает только часть выступа, а не весь лишний кусок\n"
            "views,x,y — кадр и точка, куда целиться если надо поправить "
            "(0,0 верхний левый). "
            "Если сомневаешься между ok и tiny_spot/partial, НЕЛЬЗЯ выбирать ok. "
            "note — одно короткое предложение по-русски."
        )
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}, *parts]
        try:
            text = self.chat(
                self.config.llm.vision_model,
                [{"role": "user", "content": content}],
                temperature=0.1,
            )
        except Exception:
            return {}
        return parse_mask_review(text)

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
            note = self.chat(
                self.config.llm.vision_model,
                [{"role": "user", "content": content}],
                temperature=0.2,
            ).strip()
        except Exception:
            return ""
        return self._ensure_russian(note)

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
        """Return model IDs from the OpenAI-compatible /v1/models endpoint."""
        try:
            response = self.client.models.list()
            ids = [m.id for m in response.data if getattr(m, "id", None)]
            return sorted(set(ids))
        except Exception:
            return []

    def models_status(self) -> str:
        name = llm_display_name(self.config)
        if not self.health_check():
            if name == "LM Studio":
                return "LM Studio API недоступен. Запустите Local Server в LM Studio."
            return f"{name} недоступен. Проверьте URL и API key."
        models = self.list_models()
        if not models:
            if name == "LM Studio":
                return (
                    "API отвечает, но моделей нет. Загрузите модель в LM Studio "
                    "(Chat → Load model) и обновите список."
                )
            return f"{name} отвечает, но /v1/models пуст. Проверьте ключ и каталог моделей."
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


def completion_delta_parts(delta: Any) -> list[tuple[str, str]]:
    """Split an OpenAI/LM Studio chat delta into (thinking|text, chunk) pairs."""
    from mesh_forge.agent.gpu_model import reasoning_text

    if delta is None:
        return []
    extra = getattr(delta, "model_extra", None)
    if extra is None and isinstance(delta, dict):
        extra = delta
    extra = extra or {}
    parts: list[tuple[str, str]] = []
    for field_name in ("reasoning", "reasoning_content"):
        raw = getattr(delta, field_name, None)
        if raw is None and isinstance(delta, dict):
            raw = delta.get(field_name)
        if raw is None and isinstance(extra, dict):
            raw = extra.get(field_name)
        text = reasoning_text(raw)
        if text:
            parts.append(("thinking", text))
            break
    content = getattr(delta, "content", None)
    if content is None and isinstance(delta, dict):
        content = delta.get("content")
    if isinstance(content, str) and content:
        parts.append(("text", content))
    return parts


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


def _parse_json_response(text: str) -> Any:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"operations": [], "summary": text, "parse_error": True}


def parse_mask_review(text: str) -> dict[str, Any]:
    data = _parse_json_response(text)
    if not isinstance(data, dict):
        return {}
    if data.get("parse_error"):
        return {}
    verdict = str(data.get("verdict") or "").strip().lower()
    aliases = {
        "ok": "ok",
        "good": "ok",
        "correct": "ok",
        "too_much": "too_much",
        "too much": "too_much",
        "over": "too_much",
        "too_little": "too_little",
        "too little": "too_little",
        "under": "too_little",
        "wrong": "wrong",
        "missed": "missed",
        "miss": "missed",
        "none": "missed",
        "tiny_spot": "tiny_spot",
        "tiny spot": "tiny_spot",
        "spot": "tiny_spot",
        "partial": "partial",
        "part": "partial",
    }
    verdict = aliases.get(verdict, "")
    if verdict not in {"ok", "too_much", "too_little", "wrong", "missed", "tiny_spot", "partial"}:
        return {}
    views = str(data.get("views") or "").strip().lower().split(",")[0]
    if views not in {"front", "left", "right", "back", "top", "viewer"}:
        views = ""
    try:
        conf = float(data.get("confidence", 0.5))
    except (TypeError, ValueError):
        conf = 0.5
    try:
        x = float(data.get("x", 0.5))
        y = float(data.get("y", 0.5))
    except (TypeError, ValueError):
        x, y = 0.5, 0.5
    return {
        "verdict": verdict,
        "confidence": max(0.0, min(1.0, conf)),
        "note": str(data.get("note") or "").strip(),
        "views": views,
        "x": max(0.0, min(1.0, x)),
        "y": max(0.0, min(1.0, y)),
    }


def parse_multi_view_mask_detection(text: str) -> list[dict[str, Any]]:
    data = _parse_json_response(text)
    if not isinstance(data, dict):
        return []
    if data.get("parse_error"):
        return []
    raw = data.get("observations")
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    valid_views = {"front", "left", "right", "back", "top", "viewer"}
    valid_kinds = {"protrusion", "patch", "unknown"}
    for item in raw:
        if not isinstance(item, dict):
            continue
        view = str(item.get("view") or "").strip().lower().split(",")[0]
        if view not in valid_views:
            continue
        visible = bool(item.get("visible"))
        try:
            conf = float(item.get("confidence", 0.0))
        except (TypeError, ValueError):
            conf = 0.0
        obs = {
            "view": view,
            "visible": visible,
            "confidence": max(0.0, min(1.0, conf)),
            "kind": str(item.get("kind") or "unknown").strip().lower(),
            "touchesBody": bool(item.get("touchesBody")),
            "note": str(item.get("note") or "").strip(),
        }
        if obs["kind"] not in valid_kinds:
            obs["kind"] = "unknown"
        if visible:
            try:
                x0 = max(0.0, min(1.0, float(item.get("x0"))))
                y0 = max(0.0, min(1.0, float(item.get("y0"))))
                x1 = max(0.0, min(1.0, float(item.get("x1"))))
                y1 = max(0.0, min(1.0, float(item.get("y1"))))
            except (TypeError, ValueError):
                continue
            obs["x0"], obs["x1"] = sorted((x0, x1))
            obs["y0"], obs["y1"] = sorted((y0, y1))
        out.append(obs)
    return out


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
