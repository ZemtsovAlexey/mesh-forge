#!/usr/bin/env python3
"""Shape-only Hunyuan3D-2mini inference for MeshForge Docker."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
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
    p.add_argument("--max-side", type=int, default=1280, help="Downscale long side before rembg/infer")
    p.add_argument("--no-remove-bg", action="store_true")
    p.add_argument("--no-flashvdm", action="store_true")
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def _ensure_hy3dgen_local(repo_id: str, subfolders: list[str]) -> None:
    """Materialize HY3DGEN_MODELS layout from HF hub cache so hy3dgen skips 'download' path."""
    base = Path(os.environ.get("HY3DGEN_MODELS", "~/.cache/hy3dgen")).expanduser()
    from huggingface_hub import snapshot_download

    for sub in subfolders:
        dest = base / repo_id / sub
        if dest.is_dir() and any(dest.iterdir()):
            _log(f"STAGE: weights_cached {dest}")
            continue
        _log(f"STAGE: resolve_weights {repo_id}/{sub}")
        hub_root = Path(
            snapshot_download(repo_id, allow_patterns=[f"{sub}/*"], local_files_only=False)
        )
        src = hub_root / sub
        if not src.is_dir():
            raise FileNotFoundError(f"HF snapshot missing subfolder: {src}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() or dest.is_symlink():
            continue
        try:
            dest.symlink_to(src, target_is_directory=True)
            _log(f"STAGE: weights_linked {dest} -> {src}")
        except OSError:
            import shutil

            shutil.copytree(src, dest)
            _log(f"STAGE: weights_copied {dest}")


def _downscale(image: Image.Image, max_side: int) -> Image.Image:
    w, h = image.size
    long_side = max(w, h)
    if max_side <= 0 or long_side <= max_side:
        return image
    scale = max_side / float(long_side)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    _log(f"STAGE: resize {w}x{h} -> {nw}x{nh}")
    return image.resize((nw, nh), Image.Resampling.LANCZOS)


def _opaque_count(image: Image.Image) -> tuple[int, int]:
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[-1] < 4:
        return arr.shape[0] * arr.shape[1], arr.shape[0] * arr.shape[1]
    opaque = int((arr[..., 3] > 16).sum())
    total = arr.shape[0] * arr.shape[1]
    return opaque, total


def _prepare_image(image_path: Path, *, remove_bg: bool, max_side: int) -> Image.Image:
    original = _downscale(Image.open(image_path).convert("RGBA"), max_side)
    if not remove_bg:
        return original

    _log("STAGE: rembg")
    from hy3dgen.rembg import BackgroundRemover

    cut = BackgroundRemover()(original)
    opaque, total = _opaque_count(cut)
    # Empty / almost-empty matte → empty volume / FlashVDM crash
    if opaque < max(64, total // 500):
        _log(
            f"WARN: rembg produced nearly empty mask ({opaque}/{total} px); "
            "using original image"
        )
        return original
    _log(f"STAGE: rembg_mask opaque={opaque}/{total}")
    return cut


def _mesh_ok(mesh) -> bool:
    if mesh is None:
        return False
    try:
        verts = getattr(mesh, "vertices", None)
        faces = getattr(mesh, "faces", None)
        if verts is None or faces is None:
            return False
        return len(verts) >= 3 and len(faces) >= 1
    except Exception:
        return False


def _generate_mesh(pipeline, image: Image.Image, args: argparse.Namespace, *, seed: int):
    generator = torch.Generator(device=args.device).manual_seed(seed)
    result = pipeline(
        image=image,
        num_inference_steps=args.steps,
        octree_resolution=args.octree_resolution,
        num_chunks=args.num_chunks,
        generator=generator,
        output_type="trimesh",
    )
    mesh = result[0] if isinstance(result, (list, tuple)) else result
    if not _mesh_ok(mesh):
        raise ValueError("empty mesh (no surface in volume)")
    return mesh


def _set_flashvdm(pipeline, enabled: bool) -> None:
    # Disable path reloads non-turbo VAE; keep device explicit when possible
    try:
        pipeline.enable_flashvdm(enabled=enabled, mc_algo="mc")
    except TypeError:
        pipeline.enable_flashvdm(enabled=enabled)


def main() -> int:
    os.environ.setdefault("U2NET_HOME", "/root/.u2net")
    Path(os.environ["U2NET_HOME"]).mkdir(parents=True, exist_ok=True)

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
    # Pre-stage both turbo and non-turbo VAE for flash on/off retries
    subs = [args.subfolder]
    if "mini" in args.subfolder:
        subs.extend(["hunyuan3d-vae-v2-mini-turbo", "hunyuan3d-vae-v2-mini"])
    else:
        subs.extend(["hunyuan3d-vae-v2-0-turbo", "hunyuan3d-vae-v2-0"])
    try:
        _ensure_hy3dgen_local(args.model_path, list(dict.fromkeys(subs)))
    except Exception as exc:
        _log(f"WARN: local weight prepare failed: {exc}")

    from hy3dgen.shapegen.pipelines import Hunyuan3DDiTFlowMatchingPipeline

    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        args.model_path,
        subfolder=args.subfolder,
        use_safetensors=True,
        device=args.device,
    )
    prefer_flash = not args.no_flashvdm
    if prefer_flash:
        _set_flashvdm(pipeline, True)
    else:
        _set_flashvdm(pipeline, False)
    _log("STAGE: model_ready")

    _log(f"STAGE: load_image {image_path}")
    want_rembg = not args.no_remove_bg
    image_cut = _prepare_image(image_path, remove_bg=want_rembg, max_side=args.max_side)
    image_raw = _prepare_image(image_path, remove_bg=False, max_side=args.max_side)
    _log("STAGE: rembg_done")

    # Attempts: flash+rembg → vanilla+rembg → vanilla+raw → vanilla+raw+seed
    attempts: list[tuple[str, bool, Image.Image, int]] = []
    if prefer_flash:
        attempts.append(("flash+rembg", True, image_cut, args.seed))
    attempts.append(("vanilla+rembg", False, image_cut, args.seed))
    if want_rembg:
        attempts.append(("vanilla+raw", False, image_raw, args.seed))
    attempts.append(("vanilla+raw+seed2", False, image_raw, args.seed + 7))

    # Deduplicate identical (flash, image-id, seed) while keeping order
    seen: set[tuple] = set()
    uniq_attempts = []
    for label, flash, img, seed in attempts:
        key = (flash, id(img), seed)
        if key in seen:
            continue
        seen.add(key)
        uniq_attempts.append((label, flash, img, seed))

    mesh = None
    last_err: Exception | None = None
    flash_on = prefer_flash
    for i, (label, want_flash, img, seed) in enumerate(uniq_attempts, 1):
        if want_flash != flash_on:
            _log(f"STAGE: set_flashvdm {want_flash}")
            _set_flashvdm(pipeline, want_flash)
            flash_on = want_flash
        _log(
            f"STAGE: inference attempt={i}/{len(uniq_attempts)} mode={label} "
            f"steps={args.steps} octree={args.octree_resolution} chunks={args.num_chunks}"
        )
        try:
            mesh = _generate_mesh(pipeline, img, args, seed=seed)
            _log(f"STAGE: inference_ok mode={label}")
            break
        except Exception as exc:
            last_err = exc
            _log(f"WARN: attempt failed mode={label}: {exc}")
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()

    if not _mesh_ok(mesh):
        _log(f"ERROR: all decode attempts failed: {last_err}")
        return 2

    _log("STAGE: inference_done")

    _log("STAGE: export")
    obj_path = out_dir / "mesh.obj"
    mesh.export(str(obj_path))
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
