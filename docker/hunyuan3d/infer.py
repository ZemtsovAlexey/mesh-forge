#!/usr/bin/env python3
"""Shape-only Hunyuan3D-2mini inference for MeshForge Docker."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image


def _log(msg: str) -> None:
    print(msg, flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hunyuan3D-2mini shape-only → OBJ")
    p.add_argument("image", type=str, help="Input image path")
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--model-path", type=str, default="tencent/Hunyuan3D-2mini")
    p.add_argument("--subfolder", type=str, default="hunyuan3d-dit-v2-mini-turbo")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--octree-resolution", type=int, default=256)
    p.add_argument("--num-chunks", type=int, default=8000)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--no-remove-bg", action="store_true")
    p.add_argument("--no-flashvdm", action="store_true")
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    image_path = Path(args.image)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not image_path.is_file():
        _log(f"ERROR: image not found: {image_path}")
        return 1

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        _log("ERROR: CUDA is not available inside the container")
        return 1

    _log("STAGE: loading_model")
    # Import pipeline module directly — package __init__ pulls pymeshlab postprocessors
    from hy3dgen.shapegen.pipelines import Hunyuan3DDiTFlowMatchingPipeline

    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        args.model_path,
        subfolder=args.subfolder,
        use_safetensors=True,
        device=args.device,
    )
    if not args.no_flashvdm:
        # 'mc' uses skimage — no diso CUDA build needed
        pipeline.enable_flashvdm(mc_algo="mc")
    _log("STAGE: model_ready")

    _log(f"STAGE: load_image {image_path}")
    image = Image.open(image_path).convert("RGBA")
    if not args.no_remove_bg:
        _log("STAGE: rembg")
        from hy3dgen.rembg import BackgroundRemover

        image = BackgroundRemover()(image)
    _log("STAGE: rembg_done")

    _log(
        f"STAGE: inference steps={args.steps} octree={args.octree_resolution} "
        f"chunks={args.num_chunks}"
    )
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    mesh = pipeline(
        image=image,
        num_inference_steps=args.steps,
        octree_resolution=args.octree_resolution,
        num_chunks=args.num_chunks,
        generator=generator,
        output_type="trimesh",
    )[0]
    _log("STAGE: inference_done")

    _log("STAGE: export")
    obj_path = out_dir / "mesh.obj"
    mesh.export(str(obj_path))
    # Also keep a GLB for debugging / alternate viewers
    try:
        mesh.export(str(out_dir / "mesh.glb"))
    except Exception as exc:
        _log(f"WARN: glb export skipped: {exc}")
    _log(f"STAGE: export_done {obj_path}")

    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
