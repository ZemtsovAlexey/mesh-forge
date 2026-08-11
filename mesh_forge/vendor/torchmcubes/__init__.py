"""CPU fallback for torchmcubes when the native extension cannot be built.

TripoSR's MarchingCubeHelper always remaps vertices with ``[..., [2, 1, 0]]``
to match torchmcubes' axis order. skimage returns (i, j, k) already aligned to
the volume, so we pre-swizzle here; TripoSR's remap then restores the correct
axes. Without this, meshes come out flattened / lying on their side.
"""

from __future__ import annotations

import numpy as np
import torch
from skimage.measure import marching_cubes as _marching_cubes


def marching_cubes(vol: torch.Tensor, thresh: float = 0.0):
    """Match torchmcubes.marching_cubes(vol, thresh) -> (vertices, faces)."""
    if isinstance(vol, torch.Tensor):
        if vol.is_cuda:
            vol = vol.cpu()
        volume = vol.detach().float().numpy()
    else:
        volume = np.asarray(vol, dtype=np.float32)

    verts, faces, _, _ = _marching_cubes(volume, level=float(thresh), step_size=1)
    # Pre-swizzle so TripoSR's v_pos[..., [2, 1, 0]] restores volume axes.
    verts = verts[:, [2, 1, 0]].copy()
    # Match torchmcubes face winding.
    faces = faces[:, ::-1].copy()
    return (
        torch.from_numpy(verts.astype(np.float32)),
        torch.from_numpy(faces.astype(np.int64)),
    )
