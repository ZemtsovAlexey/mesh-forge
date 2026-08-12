from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
logger = logging.getLogger("mesh_forge.config")


def _find_config() -> Path:
    candidates = [
        Path(os.environ.get("MESHFORGE_CONFIG", "")),
        ROOT / "config.yaml",
    ]
    for path in candidates:
        if path and path.is_file():
            return path
    return ROOT / "config.yaml"


@dataclass
class LLMConfig:
    provider: str = "lmstudio"
    base_url: str = "http://127.0.0.1:1234/v1"
    api_key: str = "lm-studio"
    planner_model: str = "qwen2.5-7b-instruct"
    vision_model: str = "qwen2.5-vl-7b-instruct"


@dataclass
class PathsConfig:
    openscad: str = ""
    projects: str = ""


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 7860


@dataclass
class GPUConfig:
    vram_gb: int = 0
    sequential_models: bool = True


@dataclass
class ComfyUIConfig:
    enabled: bool = True
    base_url: str = "http://127.0.0.1:8188"
    workflow: str = ""
    text_to_multiview_workflow: str = ""
    multiview_to_mesh_workflow: str = ""
    image_to_mesh_workflow: str = ""
    install_dir: str = ""
    # draft = turbo models (fast); quality = full MV + SDXL base (slower, cleaner)
    quality_preset: str = "draft"
    # img2img | zero123 | off — how left/back/right views are produced from the front
    view_consistency: str = "img2img"
    checkpoint: str = "sd_xl_turbo_1.0_fp16.safetensors"
    mesh_checkpoint: str = "hunyuan3d-dit-v2-mv-turbo_fp16.safetensors"
    image_checkpoint: str = "hunyuan3d-dit-v2-mv-turbo_fp16.safetensors"
    zero123_checkpoint: str = "stable_zero123.ckpt"
    negative_prompt: str = "blurry, lowpoly, cropped, watermark, text, logo, frame"
    width: int = 768
    height: int = 768
    steps: int = 8
    cfg: float = 1.5
    view_count: int = 4
    # img2img orbit strength (side views from front)
    view_denoise: float = 0.58
    view_denoise_turbo: float = 0.72
    view_sampler: str = "euler"
    view_scheduler: str = "sgm_uniform"
    # Zero123 novel-view orbits
    zero123_width: int = 256
    zero123_height: int = 256
    zero123_steps: int = 20
    zero123_cfg: float = 3.0
    zero123_sampler: str = "euler"
    zero123_scheduler: str = "normal"
    zero123_elevation: float = 0.0
    zero123_azimuth_left: float = -90.0
    zero123_azimuth_back: float = 180.0
    zero123_azimuth_right: float = 90.0
    mesh_resolution: int = 3072
    mesh_steps: int = 20
    mesh_cfg: float = 4.0
    mesh_guidance: float = 3.5
    mesh_octree_resolution: int = 256
    mesh_num_chunks: int = 8000


@dataclass
class PhotoConfig:
    # Scale reconstructed nets to this longest-axis size (mm)
    target_height_mm: float = 160.0
    # When False: only convert ComfyUI mesh to STL (no repair/orient/scale/component filter)
    mesh_postprocess: bool = True
    # Post-process after ComfyUI reconstruction (reduces spikes/holes)
    finalize_target_faces: int = 120_000
    finalize_smooth_iters: int = 3
    finalize_min_edge_mm: float = 0.08
    finalize_close_holes: bool = True
    # 0 = off (preserve ComfyUI mesh). Only enable for badly spiked open nets.
    finalize_voxel_mm: float = 0.0


@dataclass
class AppConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    gpu: GPUConfig = field(default_factory=GPUConfig)
    comfyui: ComfyUIConfig = field(default_factory=ComfyUIConfig)
    photo: PhotoConfig = field(default_factory=PhotoConfig)
    config_path: Path = field(default_factory=_find_config)

    @property
    def projects_dir(self) -> Path:
        raw = self.paths.projects or str(ROOT / "projects")
        return Path(raw)

    @property
    def comfyui_workflow_path(self) -> Path:
        raw = self.comfyui.workflow or str(ROOT / "mesh_forge" / "workflows" / "text_to_mesh.json")
        return Path(raw)

    @property
    def comfyui_text_to_multiview_workflow_path(self) -> Path:
        raw = self.comfyui.text_to_multiview_workflow or str(ROOT / "mesh_forge" / "workflows" / "text_to_multiview.json")
        return Path(raw)

    @property
    def comfyui_multiview_to_mesh_workflow_path(self) -> Path:
        raw = self.comfyui.multiview_to_mesh_workflow or str(ROOT / "mesh_forge" / "workflows" / "multiview_to_mesh.json")
        return Path(raw)

    @property
    def comfyui_image_to_mesh_workflow_path(self) -> Path:
        raw = self.comfyui.image_to_mesh_workflow or str(ROOT / "mesh_forge" / "workflows" / "image_to_mesh.json")
        return Path(raw)

    @property
    def comfyui_text_to_front_workflow_path(self) -> Path:
        return ROOT / "mesh_forge" / "workflows" / "text_to_front.json"

    @property
    def comfyui_zero123_orbits_workflow_path(self) -> Path:
        return ROOT / "mesh_forge" / "workflows" / "zero123_orbits.json"

    def resolve(self, key: str) -> Path | None:
        value = getattr(self.paths, key, "") or ""
        return Path(value) if value else None


def load_config(path: Path | None = None) -> AppConfig:
    cfg_path = path or _find_config()
    if not cfg_path.is_file():
        config = AppConfig(config_path=cfg_path)
        config.paths.projects = str(ROOT / "projects")
        return config

    with cfg_path.open(encoding="utf-8") as f:
        data: dict[str, Any] = yaml.safe_load(f) or {}

    photo_raw = dict(data.get("photo") or {})
    photo_raw.pop("backend", None)
    comfy_raw = dict(data.get("comfyui") or {})
    known_comfy = set(ComfyUIConfig.__dataclass_fields__)
    comfy_raw = {k: v for k, v in comfy_raw.items() if k in known_comfy}
    known_photo = set(PhotoConfig.__dataclass_fields__)
    photo_raw = {k: v for k, v in photo_raw.items() if k in known_photo}
    paths_raw = dict(data.get("paths") or {})
    known_paths = set(PathsConfig.__dataclass_fields__)
    paths_raw = {k: v for k, v in paths_raw.items() if k in known_paths}

    return AppConfig(
        llm=LLMConfig(**(data.get("llm") or {})),
        paths=PathsConfig(**paths_raw),
        server=ServerConfig(**(data.get("server") or {})),
        gpu=GPUConfig(**(data.get("gpu") or {})),
        comfyui=ComfyUIConfig(**comfy_raw),
        photo=PhotoConfig(**photo_raw),
        config_path=cfg_path,
    )


def save_config(config: AppConfig) -> Path:
    """Persist config to YAML (creates file if missing)."""
    cfg_path = config.config_path
    data: dict[str, Any] = {}
    if cfg_path.is_file():
        with cfg_path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

    data["llm"] = {
        "provider": config.llm.provider,
        "base_url": config.llm.base_url,
        "api_key": config.llm.api_key,
        "planner_model": config.llm.planner_model,
        "vision_model": config.llm.vision_model,
    }
    if config.paths.projects or data.get("paths"):
        paths = dict(data.get("paths") or {})
        paths.pop("blender", None)
        paths.pop("triposr", None)
        if config.paths.openscad:
            paths["openscad"] = config.paths.openscad
        if config.paths.projects:
            paths["projects"] = config.paths.projects
        data["paths"] = paths

    data["server"] = {
        "host": config.server.host,
        "port": config.server.port,
    }
    data["gpu"] = {
        "vram_gb": config.gpu.vram_gb,
        "sequential_models": config.gpu.sequential_models,
    }
    data.pop("docker", None)
    data["comfyui"] = {
        "enabled": config.comfyui.enabled,
        "base_url": config.comfyui.base_url,
        "install_dir": config.comfyui.install_dir,
        "workflow": config.comfyui.workflow or str(config.comfyui_workflow_path),
        "text_to_multiview_workflow": config.comfyui.text_to_multiview_workflow or str(config.comfyui_text_to_multiview_workflow_path),
        "multiview_to_mesh_workflow": config.comfyui.multiview_to_mesh_workflow or str(config.comfyui_multiview_to_mesh_workflow_path),
        "image_to_mesh_workflow": config.comfyui.image_to_mesh_workflow or str(config.comfyui_image_to_mesh_workflow_path),
        "quality_preset": config.comfyui.quality_preset,
        "view_consistency": config.comfyui.view_consistency,
        "checkpoint": config.comfyui.checkpoint,
        "mesh_checkpoint": config.comfyui.mesh_checkpoint,
        "image_checkpoint": config.comfyui.image_checkpoint,
        "zero123_checkpoint": config.comfyui.zero123_checkpoint,
        "negative_prompt": config.comfyui.negative_prompt,
        "width": config.comfyui.width,
        "height": config.comfyui.height,
        "steps": config.comfyui.steps,
        "cfg": config.comfyui.cfg,
        "view_count": config.comfyui.view_count,
        "view_denoise": config.comfyui.view_denoise,
        "view_denoise_turbo": config.comfyui.view_denoise_turbo,
        "view_sampler": config.comfyui.view_sampler,
        "view_scheduler": config.comfyui.view_scheduler,
        "zero123_width": config.comfyui.zero123_width,
        "zero123_height": config.comfyui.zero123_height,
        "zero123_steps": config.comfyui.zero123_steps,
        "zero123_cfg": config.comfyui.zero123_cfg,
        "zero123_sampler": config.comfyui.zero123_sampler,
        "zero123_scheduler": config.comfyui.zero123_scheduler,
        "zero123_elevation": config.comfyui.zero123_elevation,
        "zero123_azimuth_left": config.comfyui.zero123_azimuth_left,
        "zero123_azimuth_back": config.comfyui.zero123_azimuth_back,
        "zero123_azimuth_right": config.comfyui.zero123_azimuth_right,
        "mesh_resolution": config.comfyui.mesh_resolution,
        "mesh_steps": config.comfyui.mesh_steps,
        "mesh_cfg": config.comfyui.mesh_cfg,
        "mesh_guidance": config.comfyui.mesh_guidance,
        "mesh_octree_resolution": config.comfyui.mesh_octree_resolution,
        "mesh_num_chunks": config.comfyui.mesh_num_chunks,
    }
    data["photo"] = {
        "target_height_mm": config.photo.target_height_mm,
        "mesh_postprocess": bool(config.photo.mesh_postprocess),
        "finalize_target_faces": config.photo.finalize_target_faces,
        "finalize_smooth_iters": config.photo.finalize_smooth_iters,
        "finalize_min_edge_mm": config.photo.finalize_min_edge_mm,
        "finalize_close_holes": config.photo.finalize_close_holes,
        "finalize_voxel_mm": config.photo.finalize_voxel_mm,
    }

    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
    return cfg_path


def update_llm_settings(
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    planner_model: str | None = None,
    vision_model: str | None = None,
) -> AppConfig:
    config = load_config()
    if base_url is not None:
        config.llm.base_url = base_url.rstrip("/")
        if not config.llm.base_url.endswith("/v1"):
            config.llm.base_url = config.llm.base_url + "/v1"
    if api_key is not None:
        config.llm.api_key = api_key
    if planner_model:
        config.llm.planner_model = planner_model
    if vision_model:
        config.llm.vision_model = vision_model
    save_config(config)
    return config


QUALITY_PRESETS: dict[str, dict[str, Any]] = {
    "draft": {
        "label": "Draft (turbo, fast)",
        "checkpoint": "sd_xl_turbo_1.0_fp16.safetensors",
        "mesh_checkpoint": "hunyuan3d-dit-v2-mv-turbo_fp16.safetensors",
        "image_checkpoint": "hunyuan3d-dit-v2-mv-turbo_fp16.safetensors",
        "steps": 8,
        "cfg": 1.5,
        "mesh_steps": 20,
        "mesh_cfg": 4.0,
        "mesh_guidance": 3.5,
    },
    "quality": {
        "label": "Quality (full MV + SDXL, slower)",
        "checkpoint": "sd_xl_base_1.0_0.9vae.safetensors",
        "mesh_checkpoint": "hunyuan3d-dit-v2-mv_fp16.safetensors",
        "image_checkpoint": "hunyuan3d-dit-v2-mv_fp16.safetensors",
        "steps": 28,
        "cfg": 5.5,
        "mesh_steps": 30,
        "mesh_cfg": 5.0,
        "mesh_guidance": 5.0,
    },
}

CHECKPOINT_DOWNLOAD_URLS: dict[str, str] = {
    "sd_xl_turbo_1.0_fp16.safetensors": (
        "https://huggingface.co/stabilityai/sdxl-turbo/resolve/main/sd_xl_turbo_1.0_fp16.safetensors"
    ),
    "sd_xl_base_1.0_0.9vae.safetensors": (
        "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/"
        "sd_xl_base_1.0_0.9vae.safetensors"
    ),
    "hunyuan3d-dit-v2-mv-turbo_fp16.safetensors": (
        "https://huggingface.co/Comfy-Org/hunyuan3D_2.0_repackaged/resolve/main/split_files/"
        "hunyuan3d-dit-v2-mv-turbo_fp16.safetensors"
    ),
    "hunyuan3d-dit-v2-mv_fp16.safetensors": (
        "https://huggingface.co/Comfy-Org/hunyuan3D_2.0_repackaged/resolve/main/split_files/"
        "hunyuan3d-dit-v2-mv_fp16.safetensors"
    ),
    "stable_zero123.ckpt": (
        "https://huggingface.co/stabilityai/stable-zero123/resolve/main/stable_zero123.ckpt"
    ),
}

VIEW_CONSISTENCY_MODES: dict[str, dict[str, str]] = {
    "img2img": {
        "label": "img2img — быстро, без лишних моделей",
    },
    "zero123": {
        "label": "Zero123 — орбита от front (~9GB при первом выборе)",
    },
    "off": {
        "label": "Off — только front, single-view mesh",
    },
}


def apply_quality_preset(config: AppConfig, preset: str) -> AppConfig:
    key = (preset or "draft").strip().lower()
    if key not in QUALITY_PRESETS:
        raise ValueError(f"Unknown quality preset: {preset}. Use draft|quality")
    values = QUALITY_PRESETS[key]
    config.comfyui.quality_preset = key
    for field_name, value in values.items():
        if field_name == "label":
            continue
        setattr(config.comfyui, field_name, value)
    return config


def _comfyui_checkpoint_folders_from_api(base_url: str) -> list[Path]:
    """Ask a running ComfyUI for checkpoint folders via /experiment/models."""
    import httpx

    base = (base_url or "").rstrip("/")
    if not base:
        return []
    for path in ("/experiment/models", "/api/experiment/models"):
        try:
            with httpx.Client(timeout=3.0) as client:
                response = client.get(f"{base}{path}")
            if response.status_code != 200:
                continue
            payload = response.json()
            if not isinstance(payload, list):
                continue
            folders: list[Path] = []
            for entry in payload:
                if not isinstance(entry, dict) or entry.get("name") != "checkpoints":
                    continue
                raw_folders = entry.get("folders") or []
                if not isinstance(raw_folders, list):
                    continue
                for folder in raw_folders:
                    text = str(folder or "").strip()
                    if text:
                        folders.append(Path(text))
            if folders:
                return folders
        except Exception:
            continue
    return []


def comfyui_checkpoints_dir(config: AppConfig | None = None) -> Path | None:
    """Resolve checkpoints dir: live ComfyUI API first, then install_dir layout."""
    cfg = config or load_config()

    api_folders = _comfyui_checkpoint_folders_from_api(cfg.comfyui.base_url)
    for folder in api_folders:
        if folder.is_dir():
            return folder
    if api_folders:
        return api_folders[0]

    install = (cfg.comfyui.install_dir or "").strip()
    if not install:
        return None
    root = Path(install)
    candidates = [
        root / "models" / "checkpoints",
        root.parent / "ComfyUI" / "models" / "checkpoints",
        root / "ComfyUI" / "models" / "checkpoints",
    ]
    for path in candidates:
        if path.is_dir():
            return path
    return candidates[0]


def missing_comfyui_checkpoints(config: AppConfig | None = None) -> list[str]:
    cfg = config or load_config()
    ckpt_dir = comfyui_checkpoints_dir(cfg)
    if ckpt_dir is None:
        return []
    needed = {
        cfg.comfyui.checkpoint,
        cfg.comfyui.mesh_checkpoint,
        cfg.comfyui.image_checkpoint,
    }
    if (cfg.comfyui.view_consistency or "").strip().lower() == "zero123":
        needed.add(cfg.comfyui.zero123_checkpoint or "stable_zero123.ckpt")
    missing = []
    for name in sorted(n for n in needed if n):
        if not (ckpt_dir / name).is_file():
            missing.append(name)
    return missing


def download_comfyui_checkpoints(
    names: list[str] | None = None,
    *,
    config: AppConfig | None = None,
) -> dict[str, Any]:
    """Download missing ComfyUI checkpoints into models/checkpoints."""
    import httpx

    cfg = config or load_config()
    ckpt_dir = comfyui_checkpoints_dir(cfg)
    if ckpt_dir is None:
        raise RuntimeError(
            "comfyui.install_dir is not set — cannot download checkpoints. "
            "Set it in config.yaml (Desktop: Documents/ComfyUI; portable: .../ComfyUI) "
            "or run scripts/setup-comfyui.ps1."
        )
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    wanted = list(names) if names is not None else missing_comfyui_checkpoints(cfg)
    downloaded: list[str] = []
    skipped: list[str] = []
    errors: list[str] = []

    for name in wanted:
        dest = ckpt_dir / name
        if dest.is_file() and dest.stat().st_size > 1_000_000:
            skipped.append(name)
            continue
        url = CHECKPOINT_DOWNLOAD_URLS.get(name)
        if not url:
            errors.append(f"{name}: no download URL configured")
            continue
        part = dest.with_suffix(dest.suffix + ".part")
        logger.info("Downloading checkpoint %s -> %s", name, dest)
        try:
            with httpx.stream("GET", url, follow_redirects=True, timeout=None) as response:
                response.raise_for_status()
                total = int(response.headers.get("content-length") or 0)
                done = 0
                with part.open("wb") as out:
                    for chunk in response.iter_bytes(1024 * 1024):
                        out.write(chunk)
                        done += len(chunk)
                        if total and done % (50 * 1024 * 1024) < 1024 * 1024:
                            pct = 100.0 * done / total
                            logger.info(
                                "  %s: %.0f%% (%d / %d MB)",
                                name,
                                pct,
                                done // 1_000_000,
                                total // 1_000_000,
                            )
            part.replace(dest)
            downloaded.append(name)
            logger.info("Saved checkpoint %s (%d bytes)", name, dest.stat().st_size)
        except Exception as exc:
            if part.is_file():
                part.unlink(missing_ok=True)
            errors.append(f"{name}: {exc}")
            logger.exception("Failed to download %s", name)

    return {
        "checkpoint_dir": str(ckpt_dir),
        "downloaded": downloaded,
        "skipped": skipped,
        "errors": errors,
        "missing_checkpoints": missing_comfyui_checkpoints(cfg),
    }


def apply_view_consistency(config: AppConfig, mode: str) -> AppConfig:
    key = (mode or "img2img").strip().lower()
    if key not in VIEW_CONSISTENCY_MODES:
        raise ValueError(f"Unknown view_consistency: {mode}. Use img2img|zero123|off")
    config.comfyui.view_consistency = key
    return config


def apply_generation_knobs(config: AppConfig, knobs: dict[str, Any]) -> AppConfig:
    """Apply optional sampling/orbit knobs onto comfyui config."""
    c = config.comfyui
    mapping = {
        "checkpoint": ("checkpoint", str),
        "mesh_checkpoint": ("mesh_checkpoint", str),
        "image_checkpoint": ("image_checkpoint", str),
        "zero123_checkpoint": ("zero123_checkpoint", str),
        "width": ("width", int),
        "height": ("height", int),
        "steps": ("steps", int),
        "cfg": ("cfg", float),
        "view_denoise": ("view_denoise", float),
        "view_denoise_turbo": ("view_denoise_turbo", float),
        "view_sampler": ("view_sampler", str),
        "view_scheduler": ("view_scheduler", str),
        "zero123_width": ("zero123_width", int),
        "zero123_height": ("zero123_height", int),
        "zero123_steps": ("zero123_steps", int),
        "zero123_cfg": ("zero123_cfg", float),
        "zero123_sampler": ("zero123_sampler", str),
        "zero123_scheduler": ("zero123_scheduler", str),
        "zero123_elevation": ("zero123_elevation", float),
        "zero123_azimuth_left": ("zero123_azimuth_left", float),
        "zero123_azimuth_back": ("zero123_azimuth_back", float),
        "zero123_azimuth_right": ("zero123_azimuth_right", float),
        "mesh_steps": ("mesh_steps", int),
        "mesh_cfg": ("mesh_cfg", float),
        "mesh_guidance": ("mesh_guidance", float),
        "mesh_resolution": ("mesh_resolution", int),
        "mesh_octree_resolution": ("mesh_octree_resolution", int),
        "mesh_num_chunks": ("mesh_num_chunks", int),
    }
    for key, (attr, caster) in mapping.items():
        if key not in knobs or knobs[key] is None:
            continue
        raw = knobs[key]
        if caster is str:
            value = str(raw).strip()
            if not value:
                continue
        else:
            value = caster(raw)
        setattr(c, attr, value)
    return config


def _knobs_from_config(cfg: AppConfig) -> dict[str, Any]:
    c = cfg.comfyui
    return {
        "checkpoint": c.checkpoint,
        "mesh_checkpoint": c.mesh_checkpoint,
        "image_checkpoint": c.image_checkpoint,
        "zero123_checkpoint": c.zero123_checkpoint or "stable_zero123.ckpt",
        "width": int(c.width),
        "height": int(c.height),
        "steps": int(c.steps),
        "cfg": float(c.cfg),
        "view_denoise": float(c.view_denoise),
        "view_denoise_turbo": float(c.view_denoise_turbo),
        "view_sampler": c.view_sampler or "euler",
        "view_scheduler": c.view_scheduler or "sgm_uniform",
        "zero123_width": int(c.zero123_width),
        "zero123_height": int(c.zero123_height),
        "zero123_steps": int(c.zero123_steps),
        "zero123_cfg": float(c.zero123_cfg),
        "zero123_sampler": c.zero123_sampler or "euler",
        "zero123_scheduler": c.zero123_scheduler or "normal",
        "zero123_elevation": float(c.zero123_elevation),
        "zero123_azimuth_left": float(c.zero123_azimuth_left),
        "zero123_azimuth_back": float(c.zero123_azimuth_back),
        "zero123_azimuth_right": float(c.zero123_azimuth_right),
        "mesh_steps": int(c.mesh_steps),
        "mesh_cfg": float(c.mesh_cfg),
        "mesh_guidance": float(c.mesh_guidance),
        "mesh_resolution": int(c.mesh_resolution),
        "mesh_octree_resolution": int(c.mesh_octree_resolution),
        "mesh_num_chunks": int(c.mesh_num_chunks),
    }


def update_generation_settings(
    *,
    quality_preset: str,
    view_consistency: str | None = None,
    mesh_postprocess: bool | None = None,
    knobs: dict[str, Any] | None = None,
) -> AppConfig:
    config = load_config()
    apply_quality_preset(config, quality_preset)
    if view_consistency is not None:
        apply_view_consistency(config, view_consistency)
    if mesh_postprocess is not None:
        config.photo.mesh_postprocess = bool(mesh_postprocess)
    if knobs:
        # Knobs applied AFTER preset so manual overrides win.
        apply_generation_knobs(config, knobs)
    save_config(config)
    return config


def generation_settings_payload(
    config: AppConfig | None = None,
    *,
    download_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = config or load_config()
    preset = (cfg.comfyui.quality_preset or "draft").lower()
    if preset not in QUALITY_PRESETS:
        preset = "draft"
    view_mode = (cfg.comfyui.view_consistency or "img2img").lower()
    if view_mode not in VIEW_CONSISTENCY_MODES:
        view_mode = "img2img"
    knobs = _knobs_from_config(cfg)
    payload = {
        "quality_preset": preset,
        "view_consistency": view_mode,
        "mesh_postprocess": bool(cfg.photo.mesh_postprocess),
        "view_modes": {
            key: {"label": val["label"]} for key, val in VIEW_CONSISTENCY_MODES.items()
        },
        "presets": {
            key: {
                "label": val["label"],
                "checkpoint": val["checkpoint"],
                "mesh_checkpoint": val["mesh_checkpoint"],
                "image_checkpoint": val.get("image_checkpoint") or val["mesh_checkpoint"],
                "steps": val["steps"],
                "cfg": val["cfg"],
                "mesh_steps": val["mesh_steps"],
                "mesh_cfg": val["mesh_cfg"],
                "mesh_guidance": val["mesh_guidance"],
            }
            for key, val in QUALITY_PRESETS.items()
        },
        "knobs": knobs,
        "active": {
            "checkpoint": cfg.comfyui.checkpoint,
            "mesh_checkpoint": cfg.comfyui.mesh_checkpoint,
            "image_checkpoint": cfg.comfyui.image_checkpoint,
            "zero123_checkpoint": cfg.comfyui.zero123_checkpoint,
            "view_consistency": view_mode,
            "mesh_postprocess": bool(cfg.photo.mesh_postprocess),
            "steps": cfg.comfyui.steps,
            "cfg": cfg.comfyui.cfg,
            "mesh_steps": cfg.comfyui.mesh_steps,
            "mesh_cfg": cfg.comfyui.mesh_cfg,
            "mesh_guidance": cfg.comfyui.mesh_guidance,
            "knobs": knobs,
        },
        "missing_checkpoints": missing_comfyui_checkpoints(cfg),
        "downloaded_checkpoints": list((download_report or {}).get("downloaded") or []),
        "download_errors": list((download_report or {}).get("errors") or []),
    }
    return payload
