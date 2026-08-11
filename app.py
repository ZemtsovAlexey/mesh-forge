"""MeshForge Gradio UI entry point."""

from __future__ import annotations

import shutil
import tempfile
import traceback
from pathlib import Path

import gradio as gr

from mesh_forge.config import load_config, update_llm_settings
from mesh_forge.backends.lmstudio import LMStudioClient
from mesh_forge.manifest import ProjectManifest, create_project, list_projects
from mesh_forge.mesh_qc import analyze_mesh, is_print_ready
from mesh_forge.orchestrator import Orchestrator
from mesh_forge.render import render_mesh_preview

orch = Orchestrator()
cfg = load_config()


def _project_choices() -> list[tuple[str, str]]:
    return [(f"{p.name} ({p.id[:8]})", p.id) for p in list_projects()]


def _load_manifest(project_id: str | None) -> ProjectManifest | None:
    if not project_id:
        return None
    return ProjectManifest.load(project_id)


def refresh_projects():
    choices = _project_choices()
    ids = [c[1] for c in choices]
    return gr.update(choices=choices, value=ids[0] if ids else None)


def on_create_project(name: str):
    if not name.strip():
        return refresh_projects(), "Enter project name"
    p = create_project(name.strip())
    choices = _project_choices()
    return gr.update(choices=choices, value=p.id), f"Created: {p.name}"


def _preview_for(project_id_or_manifest: str | ProjectManifest | None):
    if isinstance(project_id_or_manifest, ProjectManifest):
        manifest = project_id_or_manifest
    else:
        manifest = _load_manifest(project_id_or_manifest)
    if not manifest:
        return None, "No project selected"
    mesh = manifest.current_mesh_path()
    if not mesh:
        return None, "No mesh yet"
    preview = manifest.root / "preview.png"
    render_mesh_preview(mesh, preview)
    stats = analyze_mesh(mesh)
    return str(preview), stats.summary()


def run_photo(project_id, image, remove_bg, solidify):
    try:
        m = _load_manifest(project_id)
        if not m or not image:
            return None, "Select project and image"
        with tempfile.TemporaryDirectory() as tmp:
            img = Path(tmp) / "input.png"
            shutil.copy(image, img)
            m, msg = orch.create_photo(m, img, remove_bg=remove_bg, solidify_mm=float(solidify or 0))
        return _preview_for(m)[0], msg
    except Exception as e:
        return None, f"Error: {e}\n{traceback.format_exc()[-800:]}"


def run_scan(project_id, scan_file, mode, smooth, solidify):
    try:
        m = _load_manifest(project_id)
        if not m or not scan_file:
            return None, "Select project and scan file"
        m, msg = orch.create_scan(
            m, Path(scan_file), mode=mode,
            smooth_iters=int(smooth), solidify_mm=float(solidify or 0),
        )
        return _preview_for(m)[0], msg
    except Exception as e:
        return None, f"Error: {e}\n{traceback.format_exc()[-800:]}"


def run_text(project_id, prompt, mode):
    try:
        m = _load_manifest(project_id)
        if not m or not prompt.strip():
            return None, "Select project and enter prompt"
        m, msg = orch.create_text(m, prompt.strip(), mode=mode)
        return _preview_for(m)[0], msg
    except Exception as e:
        return None, f"Error: {e}\n{traceback.format_exc()[-800:]}"


def run_edit_text(project_id, instruction, solidify):
    try:
        m = _load_manifest(project_id)
        if not m or not instruction.strip():
            return None, "Select project and instruction"
        m, msg = orch.edit_text(m, instruction.strip(), apply_solidify=float(solidify or 0))
        return _preview_for(m)[0], msg
    except Exception as e:
        return None, f"Error: {e}\n{traceback.format_exc()[-800:]}"


def run_edit_photo(project_id, instruction, ref_image, solidify):
    try:
        m = _load_manifest(project_id)
        if not m:
            return None, "Select project"
        ref = Path(ref_image) if ref_image else None
        m, msg = orch.edit_photo(m, instruction.strip() or "match reference", ref)
        return _preview_for(m)[0], msg
    except Exception as e:
        return None, f"Error: {e}\n{traceback.format_exc()[-800:]}"


def show_history(project_id):
    m = _load_manifest(project_id)
    if not m:
        return "No project"
    lines = [f"# {m.name} — v{m.current_version}", ""]
    for v in m.versions:
        lines.append(f"**v{v.version}** [{v.branch}] {v.action}")
        if v.instruction:
            lines.append(f"  > {v.instruction[:120]}")
        if v.qc:
            lines.append(f"  tris: {v.qc.get('triangle_count', '?')}, watertight: {v.qc.get('watertight')}")
    return "\n".join(lines)


def export_info(project_id):
    m = _load_manifest(project_id)
    if not m:
        return "No project", None
    mesh = m.current_mesh_path()
    if not mesh:
        return "No mesh to export", None
    stats = analyze_mesh(mesh)
    ready = is_print_ready(stats)
    report = stats.summary() + f"\n\nPrint ready: {'YES' if ready else 'NO — fix before slicing'}"
    return report, str(mesh)


def _llm_client_for(base_url: str, api_key: str) -> LMStudioClient:
    from mesh_forge.config import AppConfig, LLMConfig

    url = (base_url or "http://127.0.0.1:1234/v1").strip().rstrip("/")
    if not url.endswith("/v1"):
        url += "/v1"
    tmp = AppConfig(llm=LLMConfig(base_url=url, api_key=api_key or "lm-studio"))
    return LMStudioClient(tmp)


def refresh_llm_models(base_url: str, api_key: str):
    client = _llm_client_for(base_url, api_key)
    status = client.models_status()
    models = client.list_models()
    if not models:
        return (
            gr.update(choices=[], value=None),
            gr.update(choices=[], value=None),
            status,
        )
    cfg = load_config()
    planner = cfg.llm.planner_model if cfg.llm.planner_model in models else models[0]
    vision = cfg.llm.vision_model if cfg.llm.vision_model in models else models[0]
    return (
        gr.update(choices=models, value=planner),
        gr.update(choices=models, value=vision),
        status,
    )


def save_llm_settings(base_url: str, api_key: str, planner: str, vision: str):
    try:
        if not planner or not vision:
            return "Выберите модели из списка (нажмите «Обновить список»).", orch.status_text()
        update_llm_settings(
            base_url=base_url,
            api_key=api_key or "lm-studio",
            planner_model=planner,
            vision_model=vision,
        )
        orch.reload_config()
        return orch.llm.models_status(), orch.status_text()
    except Exception as e:
        return f"Ошибка сохранения: {e}", orch.status_text()


def load_llm_settings_ui():
    cfg = load_config()
    client = LMStudioClient(cfg)
    models = client.list_models()
    status = client.models_status()
    planner = cfg.llm.planner_model if cfg.llm.planner_model in models else (models[0] if models else None)
    vision = cfg.llm.vision_model if cfg.llm.vision_model in models else (models[0] if models else None)
    return (
        cfg.llm.base_url,
        cfg.llm.api_key,
        gr.update(choices=models, value=planner),
        gr.update(choices=models, value=vision),
        status,
    )


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="MeshForge", theme=gr.themes.Soft()) as app:
        gr.Markdown("# MeshForge\nФото / скан / текст → STL для 3D-печати")

        with gr.Row():
            status = gr.Textbox(label="System status", value=orch.status_text(), interactive=False, scale=2)
            gr.Button("Refresh status").click(lambda: orch.status_text(), outputs=status)

        with gr.Tab("Projects"):
            with gr.Row():
                new_name = gr.Textbox(label="New project name", placeholder="Phone stand")
                create_btn = gr.Button("Create project", variant="primary")
            project_dd = gr.Dropdown(label="Active project", choices=_project_choices(), value=None)
            create_msg = gr.Textbox(label="Message", interactive=False)
            create_btn.click(on_create_project, [new_name], [project_dd, create_msg])
            refresh_btn = gr.Button("Refresh list")
            refresh_btn.click(refresh_projects, outputs=project_dd)

        preview_img = gr.Image(label="Preview", type="filepath")
        log_out = gr.Textbox(label="Log", lines=6, interactive=False)

        with gr.Tab("Create — Photo"):
            photo_in = gr.Image(label="Photo", type="filepath")
            photo_rmbg = gr.Checkbox(label="Remove background", value=True)
            photo_solid = gr.Slider(0, 5, value=0, step=0.5, label="Solidify walls (mm), 0=off")
            gr.Button("Generate", variant="primary").click(
                run_photo, [project_dd, photo_in, photo_rmbg, photo_solid], [preview_img, log_out],
            )

        with gr.Tab("Create — Scan"):
            scan_in = gr.File(label="STL / OBJ", file_types=[".stl", ".obj"])
            scan_mode = gr.Radio(["light", "rebuild"], value="light", label="Cleanup mode")
            scan_smooth = gr.Slider(0, 5, value=1, step=1, label="Smooth iterations")
            scan_solid = gr.Slider(0, 5, value=0, step=0.5, label="Solidify (mm)")
            gr.Button("Clean scan", variant="primary").click(
                run_scan, [project_dd, scan_in, scan_mode, scan_smooth, scan_solid], [preview_img, log_out],
            )

        with gr.Tab("Create — Text"):
            text_prompt = gr.Textbox(label="Description", lines=3, placeholder="Box 80x50x30 mm with 6mm hole")
            text_mode = gr.Radio(["mechanical", "organic"], value="mechanical", label="Mode")
            gr.Button("Generate", variant="primary").click(
                run_text, [project_dd, text_prompt, text_mode], [preview_img, log_out],
            )

        with gr.Tab("Edit"):
            edit_instr = gr.Textbox(label="Text instruction", lines=2, placeholder="Height 100 mm, thicker base")
            edit_ref = gr.Image(label="Reference photo (optional)", type="filepath")
            edit_solid = gr.Slider(0, 5, value=0, step=0.5, label="Solidify after edit (mm)")
            with gr.Row():
                gr.Button("Apply text edit").click(
                    run_edit_text, [project_dd, edit_instr, edit_solid], [preview_img, log_out],
                )
                gr.Button("Apply photo edit").click(
                    run_edit_photo, [project_dd, edit_instr, edit_ref, edit_solid], [preview_img, log_out],
                )

        with gr.Tab("History"):
            hist_btn = gr.Button("Show history")
            hist_out = gr.Markdown()
            hist_btn.click(show_history, project_dd, hist_out)

        with gr.Tab("Export"):
            export_btn = gr.Button("QC report + download path")
            export_report = gr.Textbox(label="QC report", lines=10, interactive=False)
            export_path = gr.File(label="Current STL", interactive=False)
            export_btn.click(export_info, project_dd, [export_report, export_path])

        with gr.Tab("Settings — LLM"):
            gr.Markdown(
                "### LM Studio\n"
                "Загрузите модель в LM Studio (Chat → **Load model**), "
                "запустите **Local Server** (порт 1234), затем нажмите **Обновить список**."
            )
            llm_base_url = gr.Textbox(
                label="API URL",
                value=cfg.llm.base_url,
                placeholder="http://127.0.0.1:1234/v1",
            )
            llm_api_key = gr.Textbox(label="API key", value=cfg.llm.api_key, placeholder="lm-studio")
            with gr.Row():
                refresh_models_btn = gr.Button("Обновить список моделей", variant="secondary")
                save_llm_btn = gr.Button("Сохранить", variant="primary")
            llm_planner = gr.Dropdown(label="Planner (текст, OpenSCAD, edit)", choices=[], value=None)
            llm_vision = gr.Dropdown(label="Vision (фото edit)", choices=[], value=None)
            llm_status = gr.Textbox(label="Статус LM Studio", lines=8, interactive=False)

            refresh_models_btn.click(
                refresh_llm_models,
                [llm_base_url, llm_api_key],
                [llm_planner, llm_vision, llm_status],
            )
            save_llm_btn.click(
                save_llm_settings,
                [llm_base_url, llm_api_key, llm_planner, llm_vision],
                [llm_status, status],
            )
            app.load(
                load_llm_settings_ui,
                outputs=[llm_base_url, llm_api_key, llm_planner, llm_vision, llm_status],
            )

        project_dd.change(_preview_for, project_dd, [preview_img, log_out])

    return app


def main():
    cfg.projects_dir.mkdir(parents=True, exist_ok=True)
    app = build_ui()
    app.queue()
    app.launch(
        server_name=cfg.server.host,
        server_port=cfg.server.port,
        share=False,
    )


if __name__ == "__main__":
    main()
