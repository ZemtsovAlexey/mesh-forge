from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

from mesh_forge import progress as prog
from mesh_forge.adapters import ComfyUiClient
from mesh_forge.application.pipeline_run import (
    PipelineImage,
    PipelineRunState,
    abs_from_rel,
    copy_into,
    load_pipeline,
    rel_to_project,
    save_pipeline,
    work_dir_for,
)
from mesh_forge.application.project_service import ProjectService
from mesh_forge.application.prompt_chat import ChatMessage, PromptChatService
from mesh_forge.application.runner import _looks_like_clay_studio_view
from mesh_forge.domain import ImageArtifact, ImageSet
from mesh_forge.manifest import ProjectManifest
from mesh_forge.processing.mesh_ops import MeshProcessingService

logger = logging.getLogger("mesh_forge.stepped_pipeline")


def pipeline_payload(manifest: ProjectManifest, state: PipelineRunState | None = None) -> dict[str, Any]:
    state = state or load_pipeline(manifest)
    data = state.to_dict()
    images = []
    for img in state.images:
        if not img.path:
            continue
        images.append(
            {
                "label": img.label,
                "path": img.path,
                "stage": img.stage,
                "url": f"/api/projects/{manifest.id}/pipeline/image/{img.stage}/{img.label}",
            }
        )
    data["images"] = images
    return data


def _chat_note(manifest: ProjectManifest, text: str) -> None:
    from mesh_forge.application.chat_results import post_pipeline_chat_result
    from mesh_forge.application.pipeline_run import load_pipeline

    state = load_pipeline(manifest)
    posted = post_pipeline_chat_result(manifest, state, text)
    if posted is not None:
        return
    # Fallback plain note when pipeline idle
    chat = PromptChatService()
    chat_state = chat.load(manifest)
    chat_state.messages.append(ChatMessage(role="assistant", content=text))
    chat_state.assistant_message = text
    chat.save(manifest, chat_state)


def _notebook_snap(manifest: ProjectManifest, state: PipelineRunState, *, title: str | None = None) -> None:
    try:
        from mesh_forge.application.notebook import snapshot_pipeline, snapshot_version

        snapshot_pipeline(manifest, state, title=title)
        if state.step == "done" and manifest.versions:
            last = manifest.versions[-1]
            snapshot_version(manifest, last.version, instruction=last.instruction or state.brief_en or "")
    except Exception as exc:
        logger.warning("Notebook snapshot failed: %s", exc)


def _set_images(state: PipelineRunState, manifest: ProjectManifest, paths: dict[str, Path], stage: str) -> None:
    keep = [img for img in state.images if img.stage != stage]
    for label, path in paths.items():
        keep.append(
            PipelineImage(label=label, path=rel_to_project(manifest, path), stage=stage)
        )
    order = {"front": 0, "left": 1, "back": 2, "right": 3, "photo": 4, "preview": 5}
    stage_order = {"front": 0, "views": 1, "photo": 2, "mesh": 3}
    keep.sort(key=lambda i: (stage_order.get(i.stage, 50), order.get(i.label, 50), i.label))
    state.images = keep


def _image_set_from_state(manifest: ProjectManifest, state: PipelineRunState, labels: tuple[str, ...]) -> ImageSet:
    items: list[ImageArtifact] = []
    # Prefer views-stage copies when both front+views exist.
    by_label: dict[str, PipelineImage] = {}
    for img in state.images:
        prev = by_label.get(img.label)
        if prev is None or img.stage == "views" or (prev.stage != "views" and img.stage == "front"):
            by_label[img.label] = img
    for label in labels:
        entry = by_label.get(label)
        if entry is None:
            continue
        path = abs_from_rel(manifest, entry.path)
        if path.is_file():
            items.append(ImageArtifact(path=path, label=label, role="view", stage=entry.stage))
    return ImageSet(items=items)


def _clay_ok(path: Path) -> bool:
    from mesh_forge.config import load_config

    style = (load_config().comfyui.view_style or "clay").strip().lower()
    if style == "color":
        return True
    return _looks_like_clay_studio_view(path)


def _assert_clay(path: Path, label: str) -> None:
    from mesh_forge.config import load_config

    style = (load_config().comfyui.view_style or "clay").strip().lower()
    if style == "color":
        return
    if not _clay_ok(path):
        raise ValueError(
            f"Шаг «{label}» выглядит слишком цветным (не clay/studio). "
            "Нажмите «Переделать» или включите стиль Color в настройках."
        )


def prepare_photo_preview(src: Path, dest_dir: Path, *, remove_bg: bool) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "preview.png"
    raw = dest_dir / "input.png"
    shutil.copy2(src, raw)
    if remove_bg:
        try:
            from rembg import remove
            from PIL import Image
            import io

            data = remove(raw.read_bytes())
            Image.open(io.BytesIO(data)).convert("RGBA").save(dest)
            return dest
        except Exception as exc:
            logger.warning("rembg unavailable or failed (%s); using original photo", exc)
    shutil.copy2(raw, dest)
    return dest


def start_text_front(
    manifest: ProjectManifest,
    *,
    brief_en: str,
    user_prompt: str,
    solidify_mm: float = 0.0,
) -> PipelineRunState:
    from mesh_forge.adapters import LMStudioClient

    from mesh_forge.backends.lmstudio import _looks_english, _trim_subject_prompt

    brief = (brief_en or "").strip() or (user_prompt or "").strip()
    try:
        if brief and not _looks_english(brief):
            brief = LMStudioClient().ensure_english_subject(brief)
        else:
            brief = _trim_subject_prompt(brief)
    except Exception as exc:
        logger.warning("ensure_english_subject failed: %s", exc)
        if not brief.isascii():
            raise ValueError(
                "Не удалось перевести промпт на английский. "
                "Проверьте LM Studio и повторите."
            ) from exc
    if not brief:
        raise ValueError("Пустой промпт для front")

    work = work_dir_for(manifest, "stepped")
    state = PipelineRunState(
        pipeline="text_stepped",
        step="front",
        status="idle",
        brief_en=brief,
        user_prompt=user_prompt,
        solidify_mm=solidify_mm,
        work_dir=rel_to_project(manifest, work),
        message="Генерирую front…",
    )
    save_pipeline(manifest, state)
    prog.start(manifest.id, "front")
    try:
        client = ComfyUiClient()
        views = client.generate_front(brief, work / "front", project_id=manifest.id)
        front = views.get("front")
        if front is None:
            raise RuntimeError("Front view missing")
        stored = copy_into(work / "views", front.path, "front.png")
        state.step = "front"
        state.status = "ready"
        state.quality_ok = _clay_ok(stored)
        if state.quality_ok:
            state.message = "Front готов. Нажмите «Далее» для проекций или «Переделать»."
            state.error = None
        else:
            state.message = (
                "Front слишком цветный (ожидается clay/studio). "
                "Нажмите «Переделать» — «Далее» заблокировано."
            )
            state.error = state.message
        _set_images(state, manifest, {"front": stored}, "front")
        save_pipeline(manifest, state)
        _notebook_snap(manifest, state, title="Front готов")
        _chat_note(manifest, state.message)
        prog.finish(manifest.id)
        return state
    except prog.OperationCancelled:
        state.status = "error"
        state.error = "Остановлено"
        state.message = "Остановлено пользователем."
        save_pipeline(manifest, state)
        prog.finish(manifest.id, ok=False, error="Остановлено")
        raise
    except Exception as exc:
        state.status = "error"
        state.error = str(exc)
        state.message = f"Ошибка front: {exc}"
        save_pipeline(manifest, state)
        prog.finish(manifest.id, ok=False, error=str(exc))
        raise


def continue_to_views(manifest: ProjectManifest) -> PipelineRunState:
    state = load_pipeline(manifest)
    if state.pipeline != "text_stepped" or state.step != "front" or state.status != "ready":
        raise ValueError("Нет утверждённого front для генерации проекций")
    if not state.quality_ok:
        raise ValueError(state.error or "Front не прошёл проверку clay — переделайте шаг")
    work = work_dir_for(manifest, "stepped")
    front_img = next((i for i in state.images if i.label == "front"), None)
    if front_img is None:
        raise ValueError("Front artifact missing")
    front_path = abs_from_rel(manifest, front_img.path)
    _assert_clay(front_path, "front")
    state.message = "Генерирую проекции…"
    save_pipeline(manifest, state)
    prog.start(manifest.id, "views")
    try:
        client = ComfyUiClient()
        views = client.generate_views_from_front(
            state.brief_en or state.user_prompt,
            front_path,
            work / "views_run",
            project_id=manifest.id,
        )
        stored: dict[str, Path] = {}
        for label in ("front", "left", "back", "right"):
            art = views.get(label)
            if art is None:
                if label == "front":
                    stored["front"] = front_path
                    continue
                raise RuntimeError(f"Missing view: {label}")
            path = copy_into(work / "views", art.path, f"{label}.png")
            stored[label] = path
        bad = [label for label, path in stored.items() if label != "front" and not _clay_ok(path)]
        # mode=off removed: orbits are always Zero123
        state.step = "views"
        state.status = "ready"
        state.quality_ok = not bad
        if state.quality_ok:
            state.message = "Проекции готовы. Нажмите «Далее» для mesh или «Переделать»."
            state.error = None
        else:
            state.message = (
                f"Проекции слишком цветные ({', '.join(bad)}). "
                "Нажмите «Переделать» — «Далее» заблокировано."
            )
            state.error = state.message
        # Keep front stage + full views set for UI cards
        front_only = {"front": stored["front"]} if "front" in stored else {}
        if front_only:
            _set_images(state, manifest, front_only, "front")
        _set_images(state, manifest, stored, "views")
        save_pipeline(manifest, state)
        _notebook_snap(manifest, state, title="Проекции готовы")
        _chat_note(manifest, state.message)
        prog.finish(manifest.id)
        return state
    except Exception as exc:
        state.status = "error"
        state.error = str(exc)
        state.message = f"Ошибка проекций: {exc}"
        save_pipeline(manifest, state)
        prog.finish(manifest.id, ok=False, error=str(exc))
        raise


def continue_to_mesh(manifest: ProjectManifest) -> PipelineRunState:
    state = load_pipeline(manifest)
    if state.pipeline == "photo_gated":
        return _finish_photo_mesh(manifest, state)
    if state.pipeline != "text_stepped" or state.step != "views" or state.status != "ready":
        raise ValueError("Нет утверждённых проекций для mesh")
    if not state.quality_ok:
        raise ValueError(state.error or "Проекции не прошли проверку clay — переделайте шаг")
    work = work_dir_for(manifest, "stepped")
    labels = ("front", "left", "back", "right")
    image_set = _image_set_from_state(manifest, state, labels)
    if len(image_set.items) < 1:
        raise ValueError("Нет изображений для реконструкции")
    for item in image_set.items:
        _assert_clay(item.path, item.label or "view")
    state.message = "Строю mesh…"
    save_pipeline(manifest, state)
    prog.start(manifest.id, "mesh")
    try:
        client = ComfyUiClient()
        raw = client.mesh_from_views(image_set, work / "mesh", project_id=manifest.id)
        final = MeshProcessingService().finalize_reconstruction(
            raw.path,
            work / "finalize",
            solidify_mm=float(state.solidify_mm or 0.0),
        )
        artifacts: list[dict[str, Any]] = []
        for item in image_set.items:
            artifacts.append(
                {
                    "path": item.path,
                    "kind": "image",
                    "label": item.label,
                    "stage": "views",
                    "source": "stepped",
                }
            )
        artifacts.append(
            {
                "path": raw.path,
                "kind": "mesh",
                "label": "mesh_raw",
                "stage": "mesh",
                "source": "stepped",
            }
        )
        artifacts.append(
            {
                "path": final,
                "kind": "mesh",
                "label": "mesh_final",
                "stage": "finalize",
                "source": "stepped",
            }
        )
        instruction = state.user_prompt or state.brief_en
        if state.user_prompt and state.brief_en and state.user_prompt != state.brief_en:
            instruction = f"user: {state.user_prompt}\ngeneration: {state.brief_en}"
        ProjectService().add_result(
            manifest,
            Path(final),
            branch="text",
            action="create",
            instruction=instruction,
            ref=", ".join(image_set.labels()),
            artifacts=artifacts,
        )
        state.step = "done"
        state.status = "ready"
        state.message = "Mesh готов. Нажмите «Открыть проект»."
        state.error = None
        save_pipeline(manifest, state)
        # Keep conversation history; only clear ready/draft so confirm does not re-fire.
        chat = PromptChatService()
        chat_state = chat.load(manifest)
        chat_state.ready = False
        chat_state.status = "done"
        chat_state.draft_prompt_en = ""
        chat_state.edit_brief_en = ""
        chat_state.questions = []
        chat.save(manifest, chat_state)
        _notebook_snap(manifest, state, title="Mesh готов")
        _chat_note(manifest, state.message)
        prog.finish(manifest.id)
        return state
    except Exception as exc:
        state.status = "error"
        state.error = str(exc)
        state.message = f"Ошибка mesh: {exc}"
        save_pipeline(manifest, state)
        prog.finish(manifest.id, ok=False, error=str(exc))
        raise


def start_photo_gate(
    manifest: ProjectManifest,
    image_paths: list[Path],
    *,
    user_prompt: str = "",
    solidify_mm: float = 0.0,
    remove_bg: bool = True,
) -> PipelineRunState:
    if not image_paths:
        raise ValueError("Нужно хотя бы одно фото")
    work = work_dir_for(manifest, "photo_gate")
    preview = prepare_photo_preview(image_paths[0], work, remove_bg=remove_bg)
    # Keep originals for mesh
    inputs_dir = work / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for idx, src in enumerate(image_paths):
        dest = copy_into(inputs_dir, src, f"image_{idx}{src.suffix or '.png'}")
        saved.append(dest)
    state = PipelineRunState(
        pipeline="photo_gated",
        step="photo",
        status="ready",
        brief_en="",
        user_prompt=user_prompt,
        solidify_mm=solidify_mm,
        remove_bg=remove_bg,
        work_dir=rel_to_project(manifest, work),
        message="Превью входа готово. Нажмите «Далее» для mesh или «Переделать».",
    )
    _set_images(state, manifest, {"preview": preview, "photo": saved[0]}, "photo")
    # stash extra paths as numbered labels
    extra = {f"input_{i}": p for i, p in enumerate(saved)}
    for label, path in extra.items():
        state.images.append(
            PipelineImage(label=label, path=rel_to_project(manifest, path), stage="photo")
        )
    save_pipeline(manifest, state)
    _notebook_snap(manifest, state, title="Фото-превью")
    _chat_note(manifest, state.message)
    return state


def _finish_photo_mesh(manifest: ProjectManifest, state: PipelineRunState) -> PipelineRunState:
    if state.step != "photo" or state.status != "ready":
        raise ValueError("Нет утверждённого фото-превью")
    work = work_dir_for(manifest, "photo_gate")
    # Prefer rembg preview as primary if present, else inputs
    preview = next((i for i in state.images if i.label == "preview"), None)
    inputs = [i for i in state.images if i.label.startswith("input_")]
    paths: list[Path] = []
    if preview:
        paths.append(abs_from_rel(manifest, preview.path))
    elif inputs:
        paths = [abs_from_rel(manifest, i.path) for i in sorted(inputs, key=lambda x: x.label)]
    else:
        photo = next((i for i in state.images if i.label == "photo"), None)
        if photo:
            paths.append(abs_from_rel(manifest, photo.path))
    if not paths:
        raise ValueError("Нет входных изображений")
    image_set = ImageSet(
        items=[
            ImageArtifact(path=p, label="front" if i == 0 else f"view_{i}", role="view", stage="photo")
            for i, p in enumerate(paths)
            if p.is_file()
        ]
    )
    state.message = "Строю mesh из фото…"
    save_pipeline(manifest, state)
    prog.start(manifest.id, "mesh")
    try:
        client = ComfyUiClient()
        raw = client.mesh_from_views(image_set, work / "mesh", project_id=manifest.id)
        final = MeshProcessingService().finalize_reconstruction(
            raw.path,
            work / "finalize",
            solidify_mm=float(state.solidify_mm or 0.0),
        )
        artifacts = [
            {
                "path": item.path,
                "kind": "image",
                "label": item.label,
                "stage": "photo",
                "source": "photo_gate",
            }
            for item in image_set.items
        ]
        artifacts.append(
            {"path": raw.path, "kind": "mesh", "label": "mesh_raw", "stage": "mesh", "source": "photo_gate"}
        )
        artifacts.append(
            {
                "path": final,
                "kind": "mesh",
                "label": "mesh_final",
                "stage": "finalize",
                "source": "photo_gate",
            }
        )
        ProjectService().add_result(
            manifest,
            Path(final),
            branch="photo",
            action="create",
            instruction=state.user_prompt or "photo → mesh",
            ref="photo",
            artifacts=artifacts,
        )
        state.step = "done"
        state.status = "ready"
        state.message = "Mesh готов. Нажмите «Открыть проект»."
        state.error = None
        save_pipeline(manifest, state)
        chat = PromptChatService()
        chat_state = chat.load(manifest)
        chat_state.ready = False
        chat_state.status = "done"
        chat_state.draft_prompt_en = ""
        chat_state.edit_brief_en = ""
        chat_state.questions = []
        chat.save(manifest, chat_state)
        _notebook_snap(manifest, state, title="Mesh готов (фото)")
        _chat_note(manifest, state.message)
        prog.finish(manifest.id)
        return state
    except Exception as exc:
        state.status = "error"
        state.error = str(exc)
        state.message = f"Ошибка mesh: {exc}"
        save_pipeline(manifest, state)
        prog.finish(manifest.id, ok=False, error=str(exc))
        raise


def redo_step(
    manifest: ProjectManifest,
    *,
    step: str,
    brief_en: str | None = None,
) -> PipelineRunState:
    state = load_pipeline(manifest)
    if brief_en and brief_en.strip():
        state.brief_en = brief_en.strip()
        save_pipeline(manifest, state)
    if state.pipeline == "photo_gated":
        # Re-prepare from saved inputs
        inputs = sorted(
            [i for i in state.images if i.label.startswith("input_")],
            key=lambda x: x.label,
        )
        if not inputs:
            raise ValueError("Нет сохранённых фото для переделки")
        paths = [abs_from_rel(manifest, i.path) for i in inputs]
        return start_photo_gate(
            manifest,
            paths,
            user_prompt=state.user_prompt,
            solidify_mm=state.solidify_mm,
            remove_bg=state.remove_bg,
        )
    target = (step or "front").strip().lower()
    if target not in {"front", "views"}:
        raise ValueError('step must be "front" or "views"')
    if target == "front" or state.step == "front":
        return start_text_front(
            manifest,
            brief_en=state.brief_en or state.user_prompt,
            user_prompt=state.user_prompt,
            solidify_mm=state.solidify_mm,
        )
    # redo views only
    state.step = "front"
    state.status = "ready"
    save_pipeline(manifest, state)
    return continue_to_views(manifest)


def continue_pipeline(manifest: ProjectManifest) -> PipelineRunState:
    state = load_pipeline(manifest)
    if state.status != "ready":
        raise ValueError(state.error or "Пайплайн не готов к продолжению")
    if state.pipeline == "photo_gated" and state.step == "photo":
        return continue_to_mesh(manifest)
    if state.pipeline == "text_stepped" and state.step == "front":
        return continue_to_views(manifest)
    if state.pipeline == "text_stepped" and state.step == "views":
        return continue_to_mesh(manifest)
    if state.step == "done":
        raise ValueError("Пайплайн уже завершён")
    raise ValueError(f"Нельзя продолжить со шага {state.step}")


def resolve_pipeline_image(manifest: ProjectManifest, stage: str, label: str) -> Path:
    state = load_pipeline(manifest)
    for img in state.images:
        if img.label == label and img.stage == stage:
            path = abs_from_rel(manifest, img.path)
            if path.is_file():
                root = manifest.root.resolve()
                resolved = path.resolve()
                if resolved != root and root not in resolved.parents:
                    raise FileNotFoundError(label)
                return resolved
    raise FileNotFoundError(f"{stage}/{label}")
