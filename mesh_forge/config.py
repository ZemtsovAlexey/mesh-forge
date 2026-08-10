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
class AppConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    gpu: GPUConfig = field(default_factory=GPUConfig)
    config_path: Path = field(default_factory=_find_config)

    @property
    def projects_dir(self) -> Path:
        raw = self.paths.projects or str(ROOT / "projects")
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
        config_path=cfg_path,
    )
