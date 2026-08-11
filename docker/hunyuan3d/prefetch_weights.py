"""Prefetch Hunyuan3D-2mini turbo weights into HF cache (no XET)."""
from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from huggingface_hub import snapshot_download

print("DOWNLOAD_START dit-turbo", flush=True)
snapshot_download(
    "tencent/Hunyuan3D-2mini",
    allow_patterns=["hunyuan3d-dit-v2-mini-turbo/*"],
)
print("DOWNLOAD_START vae-turbo", flush=True)
snapshot_download(
    "tencent/Hunyuan3D-2mini",
    allow_patterns=["hunyuan3d-vae-v2-mini-turbo/*"],
)
print("DOWNLOAD_DONE", flush=True)
