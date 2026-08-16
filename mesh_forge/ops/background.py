from __future__ import annotations

import io
from pathlib import Path


def has_alpha(path: Path) -> bool:
    from PIL import Image

    with Image.open(path) as img:
        if img.mode != "RGBA":
            return False
        return img.getchannel("A").getextrema()[0] < 255


def cut_background(src: Path, dest: Path) -> Path:
    """Write an RGBA PNG with the subject cut out. Copies src if it already has alpha."""
    from PIL import Image
    from rembg import remove

    dest.parent.mkdir(parents=True, exist_ok=True)
    if has_alpha(src):
        Image.open(src).convert("RGBA").save(dest)
        return dest
    data = remove(src.read_bytes())
    Image.open(io.BytesIO(data)).convert("RGBA").save(dest)
    return dest
