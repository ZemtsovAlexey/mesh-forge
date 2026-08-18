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


def cut_and_flatten(
    src: Path,
    dest: Path,
    *,
    background: tuple[int, int, int] = (154, 154, 154),
) -> Path:
    """Cut the subject, then paste on a flat studio fill so Hunyuan does not mesh the backdrop."""
    from PIL import Image

    cut = dest.with_name(f"{dest.stem}_alpha.png")
    cut_background(src, cut)
    img = Image.open(cut).convert("RGBA")
    canvas = Image.new("RGB", img.size, background)
    canvas.paste(img, mask=img.getchannel("A"))
    dest.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(dest)
    return dest
