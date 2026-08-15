from __future__ import annotations

import copy
import random
from typing import Any, Literal

from pydantic import BaseModel, Field

from mesh_forge.config import AppConfig, apply_quality_preset, apply_view_style, load_config


Quality = Literal["draft", "quality"]
ViewStyle = Literal["clay", "color"]
ViewName = Literal["front", "left", "back", "right"]


class ImageKnobs(BaseModel):
    seed: int | None = None
    quality: Quality | None = None
    steps: int | None = Field(default=None, ge=1, le=50)
    cfg: float | None = Field(default=None, ge=0.5, le=15)
    style: ViewStyle | None = None


class MeshGenKnobs(BaseModel):
    seed: int | None = None
    quality: Quality | None = None
    steps: int | None = Field(default=None, ge=4, le=50)
    cfg: float | None = Field(default=None, ge=1.0, le=15)
    guidance: float | None = Field(default=None, ge=1.0, le=10)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _seed(value: int | None) -> int:
    if value is None:
        return random.randint(1, 2**31 - 1)
    return int(_clamp(int(value), 1, 2**31 - 1))


def apply_image_knobs(knobs: ImageKnobs | None) -> tuple[AppConfig, dict[str, Any]]:
    cfg = copy.deepcopy(load_config())
    knobs = knobs or ImageKnobs()
    if knobs.quality:
        apply_quality_preset(cfg, knobs.quality)
    if knobs.style:
        apply_view_style(cfg, knobs.style)
    if knobs.steps is not None:
        cfg.comfyui.steps = int(_clamp(knobs.steps, 1, 50))
    if knobs.cfg is not None:
        cfg.comfyui.cfg = float(_clamp(knobs.cfg, 0.5, 15))
    seed = _seed(knobs.seed)
    echo = {
        "seed": seed,
        "quality": cfg.comfyui.quality_preset,
        "steps": cfg.comfyui.steps,
        "cfg": cfg.comfyui.cfg,
        "style": cfg.comfyui.view_style,
    }
    return cfg, echo


def apply_mesh_knobs(knobs: MeshGenKnobs | None) -> tuple[AppConfig, dict[str, Any]]:
    cfg = copy.deepcopy(load_config())
    knobs = knobs or MeshGenKnobs()
    if knobs.quality:
        apply_quality_preset(cfg, knobs.quality)
    if knobs.steps is not None:
        cfg.comfyui.mesh_steps = int(_clamp(knobs.steps, 4, 50))
    if knobs.cfg is not None:
        cfg.comfyui.mesh_cfg = float(_clamp(knobs.cfg, 1.0, 15))
    if knobs.guidance is not None:
        cfg.comfyui.mesh_guidance = float(_clamp(knobs.guidance, 1.0, 10))
    seed = _seed(knobs.seed)
    echo = {
        "seed": seed,
        "quality": cfg.comfyui.quality_preset,
        "steps": cfg.comfyui.mesh_steps,
        "cfg": cfg.comfyui.mesh_cfg,
        "guidance": cfg.comfyui.mesh_guidance,
    }
    return cfg, echo
