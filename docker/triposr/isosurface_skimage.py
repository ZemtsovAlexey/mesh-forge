# Patched TripoSR isosurface helper: use skimage marching cubes.
# skimage returns vertices in volume index order — do NOT apply the
# torchmcubes-only axis swizzle [2, 1, 0] (that flattens / layers the mesh).

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from skimage.measure import marching_cubes as sk_marching_cubes


class IsosurfaceHelper(nn.Module):
    points_range: Tuple[float, float] = (0, 1)

    @property
    def grid_vertices(self) -> torch.FloatTensor:
        raise NotImplementedError


class MarchingCubeHelper(IsosurfaceHelper):
    def __init__(self, resolution: int) -> None:
        super().__init__()
        self.resolution = resolution
        self._grid_vertices: Optional[torch.FloatTensor] = None

    @property
    def grid_vertices(self) -> torch.FloatTensor:
        if self._grid_vertices is None:
            x, y, z = (
                torch.linspace(*self.points_range, self.resolution),
                torch.linspace(*self.points_range, self.resolution),
                torch.linspace(*self.points_range, self.resolution),
            )
            x, y, z = torch.meshgrid(x, y, z, indexing="ij")
            verts = torch.cat(
                [x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)], dim=-1
            ).reshape(-1, 3)
            self._grid_vertices = verts
        return self._grid_vertices

    def forward(
        self,
        level: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        level = -level.view(self.resolution, self.resolution, self.resolution)
        vol = level.detach().float().cpu().numpy()
        verts, faces, _, _ = sk_marching_cubes(vol, level=0.0, step_size=1)
        # skimage winding is opposite of what trimesh/Three.js expect here
        faces = faces[:, ::-1].copy()
        v_pos = torch.from_numpy(verts.copy()).float()
        t_pos_idx = torch.from_numpy(faces.copy()).long()
        v_pos = v_pos / (self.resolution - 1.0)
        return v_pos.to(level.device), t_pos_idx.to(level.device)
