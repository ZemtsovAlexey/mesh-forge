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
    checkpoint: str = "sd_xl_turbo_1.0_fp16.safetensors"
    mesh_checkpoint: str = "hunyuan3d-dit-v2-mv-turbo_fp16.safetensors"
    image_checkpoint: str = "hunyuan3d-dit-v2-mv-turbo_fp16.safetensors"
    negative_prompt: str = "blurry, lowpoly, cropped, watermark, text, logo, frame"
    width: int = 768
    height: int = 768
    steps: int = 8
    cfg: float = 1.5
    view_count: int = 4
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
    # Post-process after ComfyUI reconstruction (reduces spikes/holes)
    finalize_target_faces: int = 120_000
    finalize_smooth_iters: int = 3
    finalize_min_edge_mm: float = 0.08
    finalize_close_holes: bool = True
    # If still open after repair, voxel remesh (mm). 0 = disable.
    finalize_voxel_mm: float = 1.0


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
        "checkpoint": config.comfyui.checkpoint,
        "mesh_checkpoint": config.comfyui.mesh_checkpoint,
        "image_checkpoint": config.comfyui.image_checkpoint,
        "negative_prompt": config.comfyui.negative_prompt,
        "width": config.comfyui.width,
        "height": config.comfyui.height,
        "steps": config.comfyui.steps,
        "cfg": config.comfyui.cfg,
        "view_count": config.comfyui.view_count,
        "mesh_resolution": config.comfyui.mesh_resolution,
        "mesh_steps": config.comfyui.mesh_steps,
        "mesh_cfg": config.comfyui.mesh_cfg,
        "mesh_guidance": config.comfyui.mesh_guidance,
        "mesh_octree_resolution": config.comfyui.mesh_octree_resolution,
        "mesh_num_chunks": config.comfyui.mesh_num_chunks,
    }
    data["photo"] = {
        "target_height_mm": config.photo.target_height_mm,
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


def comfyui_checkpoints_dir(config: AppConfig | None = None) -> Path | None:
    cfg = config or load_config()
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
    missing = []
    for name in sorted(needed):
        if name and not (ckpt_dir / name).is_file():
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
            "Set it in config.yaml (e.g. C:/AI/ComfyUI_windows_portable/ComfyUI)."
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


def update_generation_settings(*, quality_preset: str) -> AppConfig:
    config = load_config()
    apply_quality_preset(config, quality_preset)
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
    payload = {
        "quality_preset": preset,
        "presets": {
            key: {
                "label": val["label"],
                "checkpoint": val["checkpoint"],
                "mesh_checkpoint": val["mesh_checkpoint"],
                "steps": val["steps"],
                "cfg": val["cfg"],
                "mesh_steps": val["mesh_steps"],
            }
            for key, val in QUALITY_PRESETS.items()
        },
        "active": {
            "checkpoint": cfg.comfyui.checkpoint,
            "mesh_checkpoint": cfg.comfyui.mesh_checkpoint,
            "image_checkpoint": cfg.comfyui.image_checkpoint,
            "steps": cfg.comfyui.steps,
            "cfg": cfg.comfyui.cfg,
            "mesh_steps": cfg.comfyui.mesh_steps,
            "mesh_cfg": cfg.comfyui.mesh_cfg,
            "mesh_guidance": cfg.comfyui.mesh_guidance,
        },
        "missing_checkpoints": missing_comfyui_checkpoints(cfg),
        "downloaded_checkpoints": list((download_report or {}).get("downloaded") or []),
        "download_errors": list((download_report or {}).get("errors") or []),
    }
    return payload
