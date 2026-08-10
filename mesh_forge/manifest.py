from __future__ import annotations

import shutil
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from mesh_forge.mesh_qc import MeshStats, analyze_mesh


@dataclass
class VersionEntry:
    version: int
    branch: str
    action: str
    mesh: str
    instruction: str | None = None
    ref: str | None = None
    ops: list[dict[str, Any]] = field(default_factory=list)
    qc: dict[str, Any] | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class ProjectManifest:
    id: str
    name: str
    current_version: int = 0
    versions: list[VersionEntry] = field(default_factory=list)

    @property
    def root(self) -> Path:
        from mesh_forge.config import load_config

        return load_config().projects_dir / self.id

    def version_dir(self, version: int | None = None) -> Path:
        v = version if version is not None else self.current_version
        return self.root / "models" / f"v{v}"

    def current_mesh_path(self) -> Path | None:
        if self.current_version < 1:
            return None
        for entry in reversed(self.versions):
            if entry.version == self.current_version:
                return self.root / entry.mesh
        return None

    def save(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        data = {
            "id": self.id,
            "name": self.name,
            "current_version": self.current_version,
            "versions": [asdict(v) for v in self.versions],
        }
        with (self.root / "manifest.yaml").open("w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)

    @classmethod
    def load(cls, project_id: str) -> ProjectManifest:
        from mesh_forge.config import load_config

        path = load_config().projects_dir / project_id / "manifest.yaml"
        if not path.is_file():
            raise FileNotFoundError(f"Project not found: {project_id}")
        with path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        versions = [VersionEntry(**v) for v in data.get("versions", [])]
        return cls(
            id=data["id"],
            name=data.get("name", project_id),
            current_version=data.get("current_version", 0),
            versions=versions,
        )


def list_projects() -> list[ProjectManifest]:
    from mesh_forge.config import load_config

    projects_dir = load_config().projects_dir
    if not projects_dir.is_dir():
        return []
    result: list[ProjectManifest] = []
    for child in sorted(projects_dir.iterdir()):
        manifest_path = child / "manifest.yaml"
        if manifest_path.is_file():
            result.append(ProjectManifest.load(child.name))
    return result


def create_project(name: str) -> ProjectManifest:
    project_id = uuid.uuid4().hex[:12]
    manifest = ProjectManifest(id=project_id, name=name)
    manifest.root.mkdir(parents=True, exist_ok=True)
    (manifest.root / "refs").mkdir(exist_ok=True)
    manifest.save()
    return manifest


def add_version(
    manifest: ProjectManifest,
    mesh_src: Path,
    *,
    branch: str,
    action: str,
    instruction: str | None = None,
    ref: str | None = None,
    ops: list[dict[str, Any]] | None = None,
    ext: str = ".stl",
) -> VersionEntry:
    new_version = manifest.current_version + 1
    version_dir = manifest.version_dir(new_version)
    version_dir.mkdir(parents=True, exist_ok=True)
    dest = version_dir / f"mesh{ext}"
    shutil.copy2(mesh_src, dest)

    stats = analyze_mesh(dest)
    rel_mesh = str(dest.relative_to(manifest.root)).replace("\\", "/")
    entry = VersionEntry(
        version=new_version,
        branch=branch,
        action=action,
        mesh=rel_mesh,
        instruction=instruction,
        ref=ref,
        ops=ops or [],
        qc=stats.to_dict(),
    )
    manifest.current_version = new_version
    manifest.versions.append(entry)
    manifest.save()
    return entry
