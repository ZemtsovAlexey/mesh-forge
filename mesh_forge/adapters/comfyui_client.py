from __future__ import annotations

import copy
import json
import logging
import random
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from mesh_forge import progress as prog
from mesh_forge.config import load_config
from mesh_forge.domain import ImageArtifact, ImageSet, MeshArtifact, TextToMeshResult
from mesh_forge.runtime import get_gpu_scheduler

logger = logging.getLogger("mesh_forge.comfyui")

VIEW_LABELS = ("front", "left", "back", "right")


@dataclass
class WorkflowPack:
    text_to_multiview: Path
    multiview_to_mesh: Path
    image_to_mesh: Path
    view_outputs: dict[str, str]
    mesh_output: str
    image_mesh_output: str


class ComfyUiClient:
    def __init__(self) -> None:
        self.config = load_config()
        self.base_url = self.config.comfyui.base_url.rstrip("/")
        self._scheduler = get_gpu_scheduler()

    def health_check(self) -> bool:
        if not self.config.comfyui.enabled:
            return False
        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.get(f"{self.base_url}/system_stats")
                return response.status_code == 200
        except Exception:
            return False

    def generate_views(self, prompt: str, work_dir: Path, *, count: int = 4, project_id: str | None = None) -> ImageSet:
        if not self.config.comfyui.enabled:
            raise RuntimeError("ComfyUI is disabled in config.yaml")

        work_dir.mkdir(parents=True, exist_ok=True)
        pack = self._load_workflow_pack()
        run_id = uuid.uuid4().hex[:8]
        workflow = self._load_text_to_multiview_workflow(pack.text_to_multiview, prompt=prompt, count=count, run_id=run_id)

        try:
            with self._scheduler.acquire("ComfyUI views", project_id=project_id):
                with httpx.Client(timeout=60.0) as client:
                    history = self._submit_workflow(client, workflow)
                    return self._collect_named_images(
                        client,
                        history=history,
                        output_dir=work_dir / "views",
                        output_nodes=pack.view_outputs,
                    )
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(self._format_prompt_error(exc)) from exc
        except httpx.HTTPError as exc:
            raise RuntimeError(
                f"ComfyUI is unavailable at {self.base_url}. "
                "Start ComfyUI on the server and make sure its API is reachable."
            ) from exc

    def run_text_to_mesh(self, prompt: str, work_dir: Path, *, project_id: str, count: int = 4) -> TextToMeshResult:
        if not self.config.comfyui.enabled:
            raise RuntimeError("ComfyUI is disabled in config.yaml")

        work_dir.mkdir(parents=True, exist_ok=True)
        pack = self._load_workflow_pack()
        run_id = uuid.uuid4().hex[:8]

        try:
            with self._scheduler.acquire("ComfyUI text→mesh", project_id=project_id):
                with httpx.Client(timeout=120.0) as client:
                    prog.update(project_id, 14, "concept")
                    view_workflow = self._load_text_to_multiview_workflow(
                        pack.text_to_multiview,
                        prompt=prompt,
                        count=count,
                        run_id=run_id,
                    )
                    view_history = self._submit_workflow(client, view_workflow)
                    prog.update(project_id, 34, "views")
                    views = self._collect_named_images(
                        client,
                        history=view_history,
                        output_dir=work_dir / "views",
                        output_nodes=pack.view_outputs,
                    )
                    if len(views.items) < 4:
                        raise RuntimeError("ComfyUI produced incomplete multiview output")

                    uploaded = self._upload_views(client, views, subfolder=f"meshforge/{run_id}")
                    prog.update(project_id, 62, "mesh")
                    mesh_workflow = self._load_multiview_to_mesh_workflow(
                        pack.multiview_to_mesh,
                        uploaded_views=uploaded,
                        run_id=run_id,
                    )
                    mesh_history = self._submit_workflow(client, mesh_workflow)
                    mesh = self._collect_mesh_artifact(
                        client,
                        history=mesh_history,
                        output_dir=work_dir / "mesh",
                        output_node=pack.mesh_output,
                    )
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(self._format_prompt_error(exc)) from exc
        except httpx.HTTPError as exc:
            raise RuntimeError(
                f"ComfyUI is unavailable at {self.base_url}. "
                "Start ComfyUI on the server and make sure its API is reachable."
            ) from exc

        return TextToMeshResult(views=views, mesh=mesh)

    def run_images_to_mesh(self, images: ImageSet, work_dir: Path, *, project_id: str) -> MeshArtifact:
        if not self.config.comfyui.enabled:
            raise RuntimeError("ComfyUI is disabled in config.yaml")
        if not images:
            raise ValueError("Image reconstruction requires at least one image")

        work_dir.mkdir(parents=True, exist_ok=True)
        pack = self._load_workflow_pack()
        run_id = uuid.uuid4().hex[:8]
        assigned = self._assign_view_paths(images)

        try:
            with self._scheduler.acquire("ComfyUI images→mesh", project_id=project_id):
                with httpx.Client(timeout=120.0) as client:
                    prog.update(project_id, 20, "mesh")
                    uploaded = {
                        label: self._upload_input_image(client, path, subfolder=f"meshforge/{run_id}")
                        for label, path in assigned.items()
                    }
                    # One checkpoint for all photo paths: missing views are filled from front
                    # (same Hunyuan MV turbo used by text→multiview→mesh).
                    filled = self._fill_missing_views(uploaded)
                    workflow = self._load_multiview_to_mesh_workflow(
                        pack.multiview_to_mesh,
                        uploaded_views=filled,
                        run_id=run_id,
                    )
                    history = self._submit_workflow(client, workflow)
                    return self._collect_mesh_artifact(
                        client,
                        history=history,
                        output_dir=work_dir / "mesh",
                        output_node=pack.mesh_output,
                    )
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(self._format_prompt_error(exc)) from exc
        except httpx.HTTPError as exc:
            raise RuntimeError(
                f"ComfyUI is unavailable at {self.base_url}. "
                "Start ComfyUI on the server and make sure its API is reachable."
            ) from exc

    def _submit_workflow(self, client: httpx.Client, workflow: dict[str, Any]) -> dict[str, Any]:
        client_id = f"meshforge-{uuid.uuid4().hex[:12]}"
        response = client.post(
            f"{self.base_url}/prompt",
            json={"prompt": workflow, "client_id": client_id},
        )
        response.raise_for_status()
        prompt_id = response.json()["prompt_id"]
        return self._wait_for_history(client, prompt_id)

    def _wait_for_history(self, client: httpx.Client, prompt_id: str) -> dict[str, Any]:
        deadline = time.time() + 1800
        while time.time() < deadline:
            response = client.get(f"{self.base_url}/history/{prompt_id}")
            response.raise_for_status()
            payload = response.json()
            if payload and prompt_id in payload:
                return payload[prompt_id]
            time.sleep(1.5)
        raise TimeoutError("Timed out waiting for ComfyUI workflow output")

    def _load_workflow_pack(self) -> WorkflowPack:
        config_path = self.config.comfyui_workflow_path
        if config_path.is_file():
            data = json.loads(config_path.read_text(encoding="utf-8"))
            if "stages" in data:
                base_dir = config_path.parent
                stages = data.get("stages") or {}
                outputs = data.get("outputs") or {}
                return WorkflowPack(
                    text_to_multiview=(base_dir / stages["text_to_multiview"]).resolve(),
                    multiview_to_mesh=(base_dir / stages["multiview_to_mesh"]).resolve(),
                    image_to_mesh=(base_dir / stages.get("image_to_mesh", "image_to_mesh.json")).resolve(),
                    view_outputs={str(k): str(v) for k, v in (outputs.get("views") or {}).items()},
                    mesh_output=str(outputs.get("mesh") or ""),
                    image_mesh_output=str(outputs.get("image_mesh") or "11"),
                )
        return WorkflowPack(
            text_to_multiview=self.config.comfyui_text_to_multiview_workflow_path,
            multiview_to_mesh=self.config.comfyui_multiview_to_mesh_workflow_path,
            image_to_mesh=self.config.comfyui_image_to_mesh_workflow_path,
            view_outputs={"front": "21", "left": "22", "back": "23", "right": "24"},
            mesh_output="17",
            image_mesh_output="11",
        )

    def _load_text_to_multiview_workflow(self, workflow_path: Path, *, prompt: str, count: int, run_id: str) -> dict[str, Any]:
        if count < 4:
            raise ValueError("Text-to-mesh workflow requires 4 named views")
        workflow = self._read_workflow(workflow_path)
        negative = self.config.comfyui.negative_prompt
        base_seed = random.randint(1, 2**31 - 1)
        replacements: dict[str, Any] = {
            "__CHECKPOINT__": self.config.comfyui.checkpoint,
            "__NEGATIVE_PROMPT__": negative,
            "__WIDTH__": self.config.comfyui.width,
            "__HEIGHT__": self.config.comfyui.height,
            "__STEPS__": self.config.comfyui.steps,
            "__CFG__": self.config.comfyui.cfg,
            "__SEED_FRONT__": base_seed,
            "__SEED_LEFT__": base_seed + 11,
            "__SEED_BACK__": base_seed + 23,
            "__SEED_RIGHT__": base_seed + 37,
            "__OUTPUT_FRONT__": f"meshforge/{run_id}/front",
            "__OUTPUT_LEFT__": f"meshforge/{run_id}/left",
            "__OUTPUT_BACK__": f"meshforge/{run_id}/back",
            "__OUTPUT_RIGHT__": f"meshforge/{run_id}/right",
        }
        for label in VIEW_LABELS:
            replacements[f"__PROMPT_{label.upper()}__"] = self._build_view_prompt(prompt, label)
        return self._render_workflow(workflow, replacements)

    def _load_multiview_to_mesh_workflow(
        self,
        workflow_path: Path,
        *,
        uploaded_views: dict[str, str],
        run_id: str,
    ) -> dict[str, Any]:
        workflow = self._read_workflow(workflow_path)
        replacements: dict[str, Any] = {
            "__MESH_CHECKPOINT__": self.config.comfyui.mesh_checkpoint,
            "__FRONT_IMAGE__": uploaded_views["front"],
            "__LEFT_IMAGE__": uploaded_views["left"],
            "__BACK_IMAGE__": uploaded_views["back"],
            "__RIGHT_IMAGE__": uploaded_views["right"],
            "__MESH_RESOLUTION__": self.config.comfyui.mesh_resolution,
            "__MESH_STEPS__": self.config.comfyui.mesh_steps,
            "__MESH_CFG__": self.config.comfyui.mesh_cfg,
            "__MESH_GUIDANCE__": self.config.comfyui.mesh_guidance,
            "__MESH_OCTREE__": self.config.comfyui.mesh_octree_resolution,
            "__MESH_CHUNKS__": self.config.comfyui.mesh_num_chunks,
            "__MESH_SEED__": random.randint(1, 2**31 - 1),
            "__MESH_OUTPUT__": f"meshforge/{run_id}/mesh",
        }
        return self._render_workflow(workflow, replacements)

    def _load_image_to_mesh_workflow(
        self,
        workflow_path: Path,
        *,
        uploaded_front: str,
        run_id: str,
    ) -> dict[str, Any]:
        workflow = self._read_workflow(workflow_path)
        replacements: dict[str, Any] = {
            "__IMAGE_CHECKPOINT__": self.config.comfyui.image_checkpoint,
            "__FRONT_IMAGE__": uploaded_front,
            "__MESH_RESOLUTION__": self.config.comfyui.mesh_resolution,
            "__MESH_STEPS__": self.config.comfyui.mesh_steps,
            "__MESH_CFG__": self.config.comfyui.mesh_cfg,
            "__MESH_GUIDANCE__": self.config.comfyui.mesh_guidance,
            "__MESH_OCTREE__": self.config.comfyui.mesh_octree_resolution,
            "__MESH_CHUNKS__": self.config.comfyui.mesh_num_chunks,
            "__MESH_SEED__": random.randint(1, 2**31 - 1),
            "__MESH_OUTPUT__": f"meshforge/{run_id}/mesh",
        }
        return self._render_workflow(workflow, replacements)

    def _assign_view_paths(self, images: ImageSet) -> dict[str, Path]:
        labeled: dict[str, Path] = {}
        unlabeled: list[Path] = []
        for item in images.items:
            key = (item.label or item.path.stem).strip().lower()
            if key in VIEW_LABELS:
                labeled[key] = item.path
            else:
                unlabeled.append(item.path)
        if labeled:
            return labeled
        assigned: dict[str, Path] = {}
        for label, path in zip(VIEW_LABELS, unlabeled):
            assigned[label] = path
        if not assigned and unlabeled:
            assigned["front"] = unlabeled[0]
        return assigned

    def _fill_missing_views(self, uploaded: dict[str, str]) -> dict[str, str]:
        if "front" not in uploaded:
            raise RuntimeError("Front view is required for multiview reconstruction")
        filled = dict(uploaded)
        for label in VIEW_LABELS:
            filled.setdefault(label, uploaded["front"])
        return filled

    def _read_workflow(self, workflow_path: Path) -> dict[str, Any]:
        if not workflow_path.is_file():
            raise FileNotFoundError(f"ComfyUI workflow not found: {workflow_path}")
        return copy.deepcopy(json.loads(workflow_path.read_text(encoding="utf-8")))

    def _render_workflow(self, workflow: dict[str, Any], replacements: dict[str, Any]) -> dict[str, Any]:
        for node in workflow.values():
            inputs = node.get("inputs", {})
            for key, value in list(inputs.items()):
                if isinstance(value, str) and value in replacements:
                    inputs[key] = replacements[value]
        return workflow

    def _build_view_prompt(self, prompt: str, label: str) -> str:
        view_text = {
            "front": "front orthographic product shot",
            "left": "left orthographic side view, 90 degrees from front",
            "back": "back orthographic view, 180 degrees from front",
            "right": "right orthographic side view, 90 degrees from front",
        }[label]
        return (
            "single object, centered, isolated on white background, clean studio lighting, "
            "consistent proportions, no hands, no scene, no extra objects, "
            f"{view_text}, {prompt.strip()}"
        )

    def _collect_named_images(
        self,
        client: httpx.Client,
        *,
        history: dict[str, Any],
        output_dir: Path,
        output_nodes: dict[str, str],
    ) -> ImageSet:
        output_dir.mkdir(parents=True, exist_ok=True)
        outputs = history.get("outputs", {})
        items: list[ImageArtifact] = []
        for label, node_id in output_nodes.items():
            node_data = outputs.get(str(node_id)) or {}
            record = self._first_output_record(node_data)
            if not record:
                continue
            suffix = Path(record["filename"]).suffix.lower() or ".png"
            dest = output_dir / f"{label}{suffix}"
            self._download_output(client, record, dest)
            items.append(ImageArtifact(path=dest, label=label, role="view", stage="views"))
        return ImageSet(items=items)

    def _collect_mesh_artifact(
        self,
        client: httpx.Client,
        *,
        history: dict[str, Any],
        output_dir: Path,
        output_node: str,
    ) -> MeshArtifact:
        output_dir.mkdir(parents=True, exist_ok=True)
        outputs = history.get("outputs", {})
        node_data = outputs.get(str(output_node)) or {}
        record = self._first_output_record(node_data)
        if not record:
            raise RuntimeError("ComfyUI produced no mesh artifact")
        suffix = Path(record["filename"]).suffix.lower() or ".glb"
        dest = output_dir / f"mesh{suffix}"
        self._download_output(client, record, dest)
        return MeshArtifact(path=dest, source="comfyui", notes="ComfyUI text-to-mesh output", label="mesh_raw", stage="mesh")

    def _upload_views(self, client: httpx.Client, views: ImageSet, *, subfolder: str) -> dict[str, str]:
        uploaded: dict[str, str] = {}
        for label in VIEW_LABELS:
            artifact = views.get(label)
            if artifact is None:
                raise RuntimeError(f"Missing required view: {label}")
            uploaded[label] = self._upload_input_image(client, artifact.path, subfolder=subfolder)
        return uploaded

    def _upload_input_image(self, client: httpx.Client, image_path: Path, *, subfolder: str) -> str:
        with image_path.open("rb") as handle:
            response = client.post(
                f"{self.base_url}/upload/image",
                data={"type": "input", "subfolder": subfolder, "overwrite": "true"},
                files={"image": (image_path.name, handle, "application/octet-stream")},
            )
        response.raise_for_status()
        payload = response.json()
        name = str(payload["name"])
        uploaded_subfolder = str(payload.get("subfolder") or "")
        return f"{uploaded_subfolder}/{name}" if uploaded_subfolder else name

    def _download_output(self, client: httpx.Client, record: dict[str, Any], dest: Path) -> None:
        response = client.get(
            f"{self.base_url}/view",
            params={
                "filename": record["filename"],
                "subfolder": record.get("subfolder", ""),
                "type": record.get("type", "output"),
            },
        )
        response.raise_for_status()
        dest.write_bytes(response.content)

    def _first_output_record(self, node_data: dict[str, Any]) -> dict[str, Any] | None:
        for values in node_data.values():
            if not isinstance(values, list):
                continue
            for item in values:
                if isinstance(item, dict) and "filename" in item:
                    return item
        return None

    def _format_prompt_error(self, exc: httpx.HTTPStatusError) -> str:
        if exc.response.status_code != 400:
            return (
                f"ComfyUI request failed with HTTP {exc.response.status_code}. "
                f"Response: {exc.response.text[:500]}"
            )
        try:
            payload = exc.response.json()
        except Exception:
            return f"ComfyUI rejected the workflow: {exc.response.text[:500]}"

        node_errors = payload.get("node_errors") or {}
        for node_data in node_errors.values():
            for error in node_data.get("errors", []):
                details = str(error.get("details", "")).strip()
                if "ckpt_name" in details and self.config.comfyui.mesh_checkpoint in details:
                    return (
                        "ComfyUI is running, but the Hunyuan3D multiview checkpoint is missing. "
                        f"Required checkpoint: {self.config.comfyui.mesh_checkpoint}. "
                        "Place it into ComfyUI models/checkpoints before running text→mesh."
                    )
                if "ckpt_name" in details and self.config.comfyui.image_checkpoint in details:
                    return (
                        "ComfyUI is running, but the Hunyuan3D image-to-mesh checkpoint is missing. "
                        f"Required checkpoint: {self.config.comfyui.image_checkpoint}. "
                        "Place it into ComfyUI models/checkpoints before running photo→mesh."
                    )
                if "ckpt_name" in details and self.config.comfyui.checkpoint in details:
                    return (
                        "ComfyUI is running, but the text-to-view checkpoint is missing. "
                        f"Configured checkpoint: {self.config.comfyui.checkpoint}."
                    )
                if details == "crop":
                    return "ComfyUI workflow is invalid: CLIPVisionEncode requires an explicit crop mode."
                if "image" in details and "not in" in details:
                    return f"ComfyUI could not resolve an uploaded multiview image: {details}"
                if details:
                    return f"ComfyUI rejected the workflow: {details}"
        return f"ComfyUI rejected the workflow: {json.dumps(payload, ensure_ascii=False)[:500]}"
