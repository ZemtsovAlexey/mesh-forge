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
class ArtifactEntry:
    kind: str
    path: str
    label: str
    stage: str = ""
    source: str = ""


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
    artifacts: list[ArtifactEntry] = field(default_factory=list)
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

    def find_view_anchor(self) -> Path | None:
        """Clay/studio reconstruction views only (front preferred)."""
        view_labels = ("front", "left", "back", "right")
        for entry in reversed(self.versions):
            by_label: dict[str, Path] = {}
            for art in entry.artifacts:
                path = self.root / art.path
                if not path.is_file() or art.kind != "image":
                    continue
                label = (art.label or path.stem).lower()
                stage = (art.stage or "").lower()
                source = (art.source or "").lower()
                if label in view_labels and (stage in {"views", "view", ""} or source in {"view", "views"}):
                    by_label[label] = path
                elif label in view_labels and stage not in {"input", "reference"}:
                    by_label[label] = path
            for label in view_labels:
                if label in by_label:
                    return by_label[label]
        return None

    def find_reference_photo(self) -> Path | None:
        """User-provided reference / input photo (not reconstruction views)."""
        for entry in reversed(self.versions):
            for art in entry.artifacts:
                path = self.root / art.path
                if not path.is_file() or art.kind != "image":
                    continue
                stage = (art.stage or "").lower()
                source = (art.source or "").lower()
                label = (art.label or path.stem).lower()
                if stage in {"input", "reference"} or source in {"reference", "input"}:
                    return path
                if label.startswith("image_") or label in {"ref", "reference"}:
                    return path
        return None

    def find_anchor_image(self) -> Path | None:
        """Guided-edit img2img anchor: reconstruction views only (never colorful ref photos)."""
        return self.find_view_anchor()

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
        versions: list[VersionEntry] = []
        for raw_version in data.get("versions", []):
            payload = dict(raw_version)
            payload["artifacts"] = [
                ArtifactEntry(**artifact)
                for artifact in payload.get("artifacts", [])
            ]
            versions.append(VersionEntry(**payload))
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


def rename_project(manifest: ProjectManifest, name: str) -> ProjectManifest:
    cleaned = (name or "").strip()
    if not cleaned:
        raise ValueError("Project name is empty")
    if len(cleaned) > 120:
        raise ValueError("Project name is too long")
    manifest.name = cleaned
    manifest.save()
    return manifest


def delete_project(project_id: str) -> None:
    from mesh_forge.config import load_config

    root = load_config().projects_dir / project_id
    if not root.is_dir() or not (root / "manifest.yaml").is_file():
        raise FileNotFoundError(f"Project not found: {project_id}")
    shutil.rmtree(root)


def duplicate_project(manifest: ProjectManifest, *, name: str | None = None) -> ProjectManifest:
    """Copy project folder to a new id (mesh history + chat included)."""
    new_id = uuid.uuid4().hex[:12]
    from mesh_forge.config import load_config

    projects_dir = load_config().projects_dir
    dest = projects_dir / new_id
    if dest.exists():
        raise RuntimeError(f"Project id collision: {new_id}")
    shutil.copytree(manifest.root, dest)
    copied = ProjectManifest.load(new_id)
    # load uses folder name; ensure id field matches
    copied.id = new_id
    copied.name = (name or f"{manifest.name} (копия)").strip()[:120]
    copied.save()
    return copied


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
    artifacts: list[dict[str, Any]] | None = None,
) -> VersionEntry:
    new_version = manifest.current_version + 1
    version_dir = manifest.version_dir(new_version)
    version_dir.mkdir(parents=True, exist_ok=True)
    dest = version_dir / f"mesh{ext}"
    shutil.copy2(mesh_src, dest)

    artifact_entries: list[ArtifactEntry] = []
    for artifact in artifacts or []:
        artifact_path = Path(str(artifact["path"]))
        artifact_label = str(artifact.get("label") or artifact_path.stem)
        artifact_dir = version_dir / "artifacts"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        copied = artifact_dir / f"{artifact_label}{artifact_path.suffix.lower()}"
        if artifact_path.resolve() != copied.resolve():
            shutil.copy2(artifact_path, copied)
        rel_artifact = str(copied.relative_to(manifest.root)).replace("\\", "/")
        artifact_entries.append(
            ArtifactEntry(
                kind=str(artifact.get("kind") or "file"),
                path=rel_artifact,
                label=artifact_label,
                stage=str(artifact.get("stage") or ""),
                source=str(artifact.get("source") or ""),
            )
        )

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
        artifacts=artifact_entries,
    )
    manifest.current_version = new_version
    manifest.versions.append(entry)
    manifest.save()
    return entry
