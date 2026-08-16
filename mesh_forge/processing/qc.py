from __future__ import annotations

from pathlib import Path

from mesh_forge.mesh_qc import MeshStats, analyze_mesh, is_print_ready


class MeshQcService:
    def analyze(self, mesh_path: Path) -> MeshStats:
        return analyze_mesh(mesh_path)

    def report(self, mesh_path: Path) -> str:
        stats = self.analyze(mesh_path)
        ready = is_print_ready(stats)
        return stats.summary() + f"\n\nГотов к печати: {'да' if ready else 'нет — сначала починить'}"
