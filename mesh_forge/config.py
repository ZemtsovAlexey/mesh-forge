from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent


def _find_config() -> Path:
    candidates = [
        Path(os.environ.get("MESHFORGE_CONFIG", "")),
        ROOT / "config.yaml",
        Path("C:/AI/mesh-forge/config.yaml"),
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
    blender: str = ""
    openscad: str = ""
    triposr: str = ""
    projects: str = ""


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 7860


@dataclass
class GPUConfig:
    vram_gb: int = 8
    sequential_models: bool = True


@dataclass
class DockerConfig:
    enabled: bool = True
    triposr_image: str = "meshforge/triposr:latest"
    hf_cache: str = ""


@dataclass
class AppConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    gpu: GPUConfig = field(default_factory=GPUConfig)
    docker: DockerConfig = field(default_factory=DockerConfig)
    config_path: Path = field(default_factory=_find_config)

    @property
    def projects_dir(self) -> Path:
        raw = self.paths.projects or str(ROOT / "projects")
        return Path(raw)

    @property
    def hf_cache_dir(self) -> Path:
        raw = self.docker.hf_cache or str(ROOT / ".cache" / "huggingface")
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

    return AppConfig(
        llm=LLMConfig(**(data.get("llm") or {})),
        paths=PathsConfig(**(data.get("paths") or {})),
        server=ServerConfig(**(data.get("server") or {})),
        gpu=GPUConfig(**(data.get("gpu") or {})),
        docker=DockerConfig(**(data.get("docker") or {})),
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
        if config.paths.blender:
            paths["blender"] = config.paths.blender
        if config.paths.openscad:
            paths["openscad"] = config.paths.openscad
        if config.paths.triposr:
            paths["triposr"] = config.paths.triposr
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
    data["docker"] = {
        "enabled": config.docker.enabled,
        "triposr_image": config.docker.triposr_image,
        "hf_cache": config.docker.hf_cache or str(config.hf_cache_dir),
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
