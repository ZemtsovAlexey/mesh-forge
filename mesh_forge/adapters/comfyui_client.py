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
from mesh_forge.config import load_config, normalize_comfyui_base_url
from mesh_forge.domain import ImageArtifact, ImageSet, MeshArtifact, TextToMeshResult
from mesh_forge.runtime import get_gpu_scheduler

logger = logging.getLogger("mesh_forge.comfyui")

VIEW_LABELS = ("front", "left", "back", "right")
_MESH_SUFFIXES = {".glb", ".gltf", ".obj", ".stl", ".ply"}
_ISOLATION_SUFFIX = (
    "single isolated object, one item only, entire object fully in frame "
    "including legs and top, centered product shot, plain contrasting studio backdrop, "
    "background must not match the object, no extra copies, no crop"
)
_GEOMETRY_SUFFIX = (
    "standing on a flat ground plane, all legs or contact points on one level, "
    "no tilt, no perspective distortion, no fisheye, no dutch angle"
)
_CAMERA_BY_VIEW = {
    "front": (
        "orthographic front elevation, eye-level camera looking straight on, "
        "level horizon, object facing the camera, both sides equally visible, no three-quarter"
    ),
    "left": (
        "orthographic left profile, eye-level true side view, "
        "level horizon, 90-degree yaw from front, no three-quarter"
    ),
    "back": (
        "orthographic rear elevation, eye-level camera behind the object, "
        "level horizon, object facing away, no three-quarter"
    ),
    "right": (
        "orthographic right profile, eye-level true side view, "
        "level horizon, 90-degree yaw from front, no three-quarter"
    ),
}
_VIEW_NEGATIVE_EXTRAS = (
    "photorealistic photo, busy scene, collage, split screen, multiple subjects, "
    "inconsistent identity across views, morphing shape, disconnected floating parts, "
    "hands, people, text, watermark, logo, frame, cropped, cut off, "
    "landscape background, sky, grass, trees, fence, scenery, 2d illustration background, "
    "filament spools, sewing thread, workshop, crafts, "
    "matching background texture, wood paneled wall, furniture showroom, "
    "row of identical items, extra copies of the object, "
    "three-quarter view, 3/4 view, isometric, dutch angle, tilted horizon, leaning object, "
    "uneven legs, warped geometry, melting, perspective distortion, fisheye, "
    "high angle, low angle, bird's eye, worm's eye, diagonal view, "
    "side profile when front is required, three-quarter view when front is required"
)
_CLAY_NEGATIVE_EXTRAS = (
    "colorful plastic, painted texture, multicolored patterns, "
    "glossy materials, fabric, metallic reflections"
)


def isolation_view_prompt(subject: str, style: str = "clay", view: str = "front") -> str:
    text = (subject or "").strip().rstrip(",")
    label = (view or "front").strip().lower()
    camera = _CAMERA_BY_VIEW.get(label, _CAMERA_BY_VIEW["front"])
    extra = f"{camera}, {_GEOMETRY_SUFFIX}, {_ISOLATION_SUFFIX}"
    if (style or "clay").strip().lower() == "clay":
        extra = f"{extra}, matte clay, uniform light grey, no photoreal materials"
    return f"{text}, {extra}" if text else extra


@dataclass
class WorkflowPack:
    text_to_front: Path
    zero123_orbits: Path
    guided_edit_front: Path
    multiview_to_mesh: Path
    image_to_mesh: Path
    front_output: str
    guided_front_output: str
    orbit_outputs: dict[str, str]
    mesh_output: str
    image_mesh_output: str


class ComfyUiClient:
    def __init__(self) -> None:
        self._scheduler = get_gpu_scheduler()
        self._refresh()

    def _refresh(self) -> None:
        self.config = load_config()
        self.base_url = normalize_comfyui_base_url(self.config.comfyui.base_url)

    @staticmethod
    def probe(base_url: str, timeout: float = 8.0) -> tuple[bool, str]:
        url = normalize_comfyui_base_url(base_url)
        try:
            with httpx.Client(timeout=timeout) as client:
                response = client.get(f"{url}/system_stats")
            if response.status_code == 200:
                return True, "OK"
            return False, f"HTTP {response.status_code}"
        except Exception as exc:
            return False, str(exc)

    def health_check(self) -> bool:
        self._refresh()
        if not self.config.comfyui.enabled:
            return False
        ok, _ = self.probe(self.base_url, timeout=10.0)
        return ok

    def generate_views(
        self,
        prompt: str,
        work_dir: Path,
        *,
        count: int = 4,
        project_id: str | None = None,
        seed: int | None = None,
    ) -> ImageSet:
        """Text → front, then Zero123 left/back/right orbits."""
        self._refresh()
        if not self.config.comfyui.enabled:
            raise RuntimeError("ComfyUI is disabled in config.yaml")

        work_dir.mkdir(parents=True, exist_ok=True)
        pack = self._load_workflow_pack()
        run_id = uuid.uuid4().hex[:8]
        _ = count

        try:
            with self._scheduler.acquire("ComfyUI views", kind="comfy", project_id=project_id):
                with httpx.Client(timeout=120.0) as client:
                    front_wf = self._load_text_to_front_workflow(
                        pack.text_to_front, prompt=prompt, run_id=run_id, seed=seed
                    )
                    front_history = self._submit_workflow(client, front_wf)
                    front_views = self._collect_front_image(
                        client,
                        history=front_history,
                        output_dir=work_dir / "views",
                        output_node=pack.front_output,
                    )
                    return self._generate_zero123_orbits(
                        client,
                        pack=pack,
                        front_views=front_views,
                        work_dir=work_dir,
                        run_id=run_id,
                        seed=seed,
                    )
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(self._format_prompt_error(exc)) from exc
        except httpx.HTTPError as exc:
            raise RuntimeError(
                f"ComfyUI is unavailable at {self.base_url}. "
                "Start ComfyUI on the server and make sure its API is reachable."
            ) from exc

    def run_text_to_mesh(self, prompt: str, work_dir: Path, *, project_id: str, count: int = 4) -> TextToMeshResult:
        """Text → front + Zero123 orbits → Hunyuan mesh."""
        self._refresh()
        if not self.config.comfyui.enabled:
            raise RuntimeError("ComfyUI is disabled in config.yaml")

        work_dir.mkdir(parents=True, exist_ok=True)
        pack = self._load_workflow_pack()
        run_id = uuid.uuid4().hex[:8]
        _ = count

        try:
            with self._scheduler.acquire("ComfyUI text→mesh", kind="comfy", project_id=project_id):
                with httpx.Client(timeout=180.0) as client:
                    prog.update(project_id, 14, "concept")
                    front_wf = self._load_text_to_front_workflow(
                        pack.text_to_front, prompt=prompt, run_id=run_id
                    )
                    front_history = self._submit_workflow(client, front_wf)
                    prog.update(project_id, 34, "views")
                    views = self._collect_front_image(
                        client,
                        history=front_history,
                        output_dir=work_dir / "views",
                        output_node=pack.front_output,
                    )
                    views = self._generate_zero123_orbits(
                        client, pack=pack, front_views=views, work_dir=work_dir, run_id=run_id
                    )

                    prog.update(project_id, 62, "mesh")
                    uploaded = self._upload_views(client, views, subfolder=f"meshforge/{run_id}")
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

    def generate_front(self, prompt: str, work_dir: Path, *, project_id: str, seed: int | None = None) -> ImageSet:
        """Text → single front view."""
        self._refresh()
        if not self.config.comfyui.enabled:
            raise RuntimeError("ComfyUI is disabled in config.yaml")
        work_dir.mkdir(parents=True, exist_ok=True)
        pack = self._load_workflow_pack()
        run_id = uuid.uuid4().hex[:8]
        try:
            with self._scheduler.acquire("ComfyUI front", kind="comfy", project_id=project_id):
                with httpx.Client(timeout=180.0) as client:
                    prog.update(project_id, 20, "front")
                    front_wf = self._load_text_to_front_workflow(
                        pack.text_to_front, prompt=prompt, run_id=run_id, seed=seed
                    )
                    history = self._submit_workflow(client, front_wf)
                    views = self._collect_front_image(
                        client,
                        history=history,
                        output_dir=work_dir / "views",
                        output_node=pack.front_output,
                    )
                    if not views.items:
                        raise RuntimeError("ComfyUI produced no front view")
                    return views
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(self._format_prompt_error(exc)) from exc
        except httpx.HTTPError as exc:
            raise RuntimeError(
                f"ComfyUI is unavailable at {self.base_url}. "
                "Start ComfyUI on the server and make sure its API is reachable."
            ) from exc

    def generate_views_from_front(
        self,
        prompt: str,
        front_image: Path,
        work_dir: Path,
        *,
        project_id: str,
        seed: int | None = None,
    ) -> ImageSet:
        """Approved front → Zero123 left/back/right orbits."""
        _ = prompt
        self._refresh()
        if not self.config.comfyui.enabled:
            raise RuntimeError("ComfyUI is disabled in config.yaml")
        if not front_image.is_file():
            raise FileNotFoundError(f"Front image missing: {front_image}")
        work_dir.mkdir(parents=True, exist_ok=True)
        pack = self._load_workflow_pack()
        run_id = uuid.uuid4().hex[:8]

        try:
            with self._scheduler.acquire("ComfyUI views-from-front", kind="comfy", project_id=project_id):
                with httpx.Client(timeout=180.0) as client:
                    prog.update(project_id, 40, "views")
                    views_dir = work_dir / "views"
                    views_dir.mkdir(parents=True, exist_ok=True)
                    front_dest = views_dir / "front.png"
                    if front_image.resolve() != front_dest.resolve():
                        front_dest.write_bytes(front_image.read_bytes())
                    front_views = ImageSet(
                        items=[ImageArtifact(path=front_dest, label="front", role="view", stage="views")]
                    )
                    return self._generate_zero123_orbits(
                        client,
                        pack=pack,
                        front_views=front_views,
                        work_dir=work_dir,
                        run_id=run_id,
                        seed=seed,
                    )
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(self._format_prompt_error(exc)) from exc
        except httpx.HTTPError as exc:
            raise RuntimeError(
                f"ComfyUI is unavailable at {self.base_url}. "
                "Start ComfyUI on the server and make sure its API is reachable."
            ) from exc

    def mesh_from_views(
        self,
        views: ImageSet,
        work_dir: Path,
        *,
        project_id: str,
    ) -> MeshArtifact:
        """Approved views → Hunyuan mesh (stepped pipeline)."""
        return self.run_images_to_mesh(views, work_dir, project_id=project_id)

    def run_guided_edit(
        self,
        prompt: str,
        anchor_image: Path,
        work_dir: Path,
        *,
        project_id: str,
        count: int = 4,
    ) -> TextToMeshResult:
        """Preserve identity: img2img the front, then Zero123 orbits → mesh."""
        self._refresh()
        if not self.config.comfyui.enabled:
            raise RuntimeError("ComfyUI is disabled in config.yaml")
        if not anchor_image.is_file():
            raise FileNotFoundError(f"Guided-edit anchor missing: {anchor_image}")

        work_dir.mkdir(parents=True, exist_ok=True)
        pack = self._load_workflow_pack()
        run_id = uuid.uuid4().hex[:8]
        _ = count

        try:
            with self._scheduler.acquire("ComfyUI guided edit", kind="comfy", project_id=project_id):
                with httpx.Client(timeout=180.0) as client:
                    prog.update(project_id, 14, "guided")
                    uploaded_anchor = self._upload_input_image(
                        client, anchor_image, subfolder=f"meshforge/{run_id}"
                    )
                    edit_wf = self._load_guided_edit_front_workflow(
                        pack.guided_edit_front,
                        prompt=prompt,
                        uploaded_anchor=uploaded_anchor,
                        run_id=run_id,
                    )
                    edit_history = self._submit_workflow(client, edit_wf)
                    prog.update(project_id, 34, "views")
                    views = self._collect_front_image(
                        client,
                        history=edit_history,
                        output_dir=work_dir / "views",
                        output_node=pack.guided_front_output,
                    )
                    views = self._generate_zero123_orbits(
                        client,
                        pack=pack,
                        front_views=views,
                        work_dir=work_dir,
                        run_id=run_id,
                    )

                    prog.update(project_id, 62, "mesh")
                    uploaded = self._upload_views(client, views, subfolder=f"meshforge/{run_id}")
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

    def _load_guided_edit_front_workflow(
        self,
        workflow_path: Path,
        *,
        prompt: str,
        uploaded_anchor: str,
        run_id: str,
        seed: int | None = None,
    ) -> dict[str, Any]:
        workflow = self._read_workflow(workflow_path)
        base_seed = seed if seed is not None else random.randint(1, 2**31 - 1)
        c = self.config.comfyui
        edit_denoise = float(getattr(c, "edit_denoise", 0.28) or 0.28)
        edit_denoise = min(max(edit_denoise, 0.12), 0.55)
        negative = self._build_view_negative(c.negative_prompt)
        subject = self._normalize_subject_prompt(prompt)
        replacements: dict[str, Any] = {
            "__CHECKPOINT__": c.checkpoint,
            "__ANCHOR_IMAGE__": uploaded_anchor,
            "__NEGATIVE_PROMPT__": negative,
            "__PROMPT__": self._build_view_prompt(subject, "front"),
            "__WIDTH__": c.width,
            "__HEIGHT__": c.height,
            "__STEPS__": c.steps,
            "__CFG__": c.cfg,
            "__SAMPLER__": c.view_sampler or "euler",
            "__SCHEDULER__": c.view_scheduler or "sgm_uniform",
            "__EDIT_DENOISE__": edit_denoise,
            "__SEED__": base_seed,
            "__OUTPUT_FRONT__": f"meshforge/{run_id}/front",
        }
        logger.info(
            "guided-edit seed=%s edit_denoise=%.2f subject=%s",
            base_seed,
            edit_denoise,
            subject[:120],
        )
        return self._render_workflow(workflow, replacements)

    def _generate_zero123_orbits(
        self,
        client: httpx.Client,
        *,
        pack: WorkflowPack,
        front_views: ImageSet,
        work_dir: Path,
        run_id: str,
        seed: int | None = None,
    ) -> ImageSet:
        front = front_views.get("front")
        if front is None:
            raise RuntimeError("Front view required for Zero123 orbits")
        ckpt = self.config.comfyui.zero123_checkpoint or "stable_zero123.ckpt"
        ckpt_dir = None
        try:
            from mesh_forge.config import comfyui_checkpoints_dir

            ckpt_dir = comfyui_checkpoints_dir(self.config)
        except Exception:
            pass
        if ckpt_dir is not None and not (ckpt_dir / ckpt).is_file():
            raise RuntimeError(
                f"Zero123 checkpoint missing: {ckpt}. "
                "Run scripts/setup-comfyui.ps1 or put the file in ComfyUI/models/checkpoints."
            )
        uploaded_front = self._upload_input_image(client, front.path, subfolder=f"meshforge/{run_id}")
        orbit_wf = self._load_zero123_orbits_workflow(
            pack.zero123_orbits, uploaded_front=uploaded_front, run_id=run_id, seed=seed
        )
        orbit_history = self._submit_workflow(client, orbit_wf)
        orbits = self._collect_named_images(
            client,
            history=orbit_history,
            output_dir=work_dir / "views",
            output_nodes=pack.orbit_outputs,
        )
        items = [art for art in front_views.items if (art.label or "").lower() == "front"]
        if not items:
            items = list(front_views.items[:1])
        for label in ("left", "back", "right"):
            art = orbits.get(label)
            if art is None:
                raise RuntimeError(f"Zero123 did not produce {label} view")
            items.append(art)
        return ImageSet(items=items)

    def run_images_to_mesh(
        self,
        images: ImageSet,
        work_dir: Path,
        *,
        project_id: str,
        seed: int | None = None,
    ) -> MeshArtifact:
        self._refresh()
        if not self.config.comfyui.enabled:
            raise RuntimeError("ComfyUI is disabled in config.yaml")
        if not images:
            raise ValueError("Image reconstruction requires at least one image")

        work_dir.mkdir(parents=True, exist_ok=True)
        pack = self._load_workflow_pack()
        run_id = uuid.uuid4().hex[:8]
        assigned = self._assign_view_paths(images)

        try:
            with self._scheduler.acquire("ComfyUI images→mesh", kind="comfy", project_id=project_id):
                with httpx.Client(timeout=120.0) as client:
                    prog.update(project_id, 20, "mesh")
                    uploaded = {
                        label: self._upload_input_image(client, path, subfolder=f"meshforge/{run_id}")
                        for label, path in assigned.items()
                    }
                    unique_paths = {p.resolve() for p in assigned.values()}
                    if len(assigned) == 1 or len(unique_paths) == 1:
                        front_key = "front" if "front" in uploaded else next(iter(uploaded))
                        workflow = self._load_image_to_mesh_workflow(
                            pack.image_to_mesh,
                            uploaded_front=uploaded[front_key],
                            run_id=run_id,
                            seed=seed,
                        )
                        output_node = pack.image_mesh_output
                    else:
                        workflow = self._load_multiview_to_mesh_workflow(
                            pack.multiview_to_mesh,
                            uploaded_views=uploaded,
                            run_id=run_id,
                            seed=seed,
                        )
                        output_node = pack.mesh_output
                    history = self._submit_workflow(client, workflow)
                    return self._collect_mesh_artifact(
                        client,
                        history=history,
                        output_dir=work_dir / "mesh",
                        output_node=output_node,
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
        project_id = self._scheduler.active_project_id
        prog.raise_if_cancelled(project_id)
        response = client.post(
            f"{self.base_url}/prompt",
            json={"prompt": workflow, "client_id": client_id},
        )
        response.raise_for_status()
        prompt_id = response.json()["prompt_id"]
        return self._wait_for_history(client, prompt_id, project_id=project_id)

    def interrupt(self) -> None:
        """Ask ComfyUI to stop the currently executing prompt."""
        self._refresh()
        try:
            with httpx.Client(timeout=10.0) as client:
                client.post(f"{self.base_url}/interrupt")
        except Exception as exc:
            logger.warning("ComfyUI interrupt failed: %s", exc)

    def _wait_for_history(
        self,
        client: httpx.Client,
        prompt_id: str,
        *,
        project_id: str | None = None,
    ) -> dict[str, Any]:
        deadline = time.time() + 1800
        while time.time() < deadline:
            if prog.is_cancelled(project_id):
                try:
                    client.post(f"{self.base_url}/interrupt")
                except Exception:
                    pass
                try:
                    client.post(f"{self.base_url}/queue", json={"delete": [prompt_id]})
                except Exception:
                    pass
                if project_id:
                    prog.finish(project_id, ok=False, error="Остановлено")
                raise prog.OperationCancelled("Остановлено пользователем")
            response = client.get(f"{self.base_url}/history/{prompt_id}")
            response.raise_for_status()
            payload = response.json()
            if payload and prompt_id in payload:
                prog.raise_if_cancelled(project_id)
                entry = payload[prompt_id]
                status = entry.get("status") if isinstance(entry, dict) else None
                if isinstance(status, dict) and status.get("completed") is False:
                    time.sleep(1.0)
                    continue
                return entry
            time.sleep(1.0)
        raise TimeoutError("Timed out waiting for ComfyUI workflow output")

    def _load_workflow_pack(self) -> WorkflowPack:
        config_path = self.config.comfyui_workflow_path
        workflows_dir = Path(__file__).resolve().parent.parent / "workflows"
        default_text_to_front = workflows_dir / "text_to_front.json"
        default_zero123 = workflows_dir / "zero123_orbits.json"
        default_guided = workflows_dir / "guided_edit_front.json"
        orbit_defaults = {"left": "22", "back": "23", "right": "24"}
        if config_path.is_file():
            data = json.loads(config_path.read_text(encoding="utf-8"))
            if "stages" in data:
                base_dir = config_path.parent
                stages = data.get("stages") or {}
                outputs = data.get("outputs") or {}
                front_path = stages.get("text_to_front") or "text_to_front.json"
                zero_path = stages.get("zero123_orbits") or "zero123_orbits.json"
                guided_path = (
                    stages.get("guided_edit_front")
                    or stages.get("guided_edit_multiview")
                    or "guided_edit_front.json"
                )
                orbit_outputs = {
                    str(k): str(v)
                    for k, v in (outputs.get("orbits") or orbit_defaults).items()
                }
                return WorkflowPack(
                    text_to_front=(base_dir / front_path).resolve(),
                    zero123_orbits=(base_dir / zero_path).resolve(),
                    guided_edit_front=(base_dir / guided_path).resolve(),
                    multiview_to_mesh=(base_dir / stages["multiview_to_mesh"]).resolve(),
                    image_to_mesh=(base_dir / stages.get("image_to_mesh", "image_to_mesh.json")).resolve(),
                    front_output=str(outputs.get("front") or "7"),
                    guided_front_output=str(outputs.get("guided_front") or "21"),
                    orbit_outputs=orbit_outputs,
                    mesh_output=str(outputs.get("mesh") or ""),
                    image_mesh_output=str(outputs.get("image_mesh") or "11"),
                )
        return WorkflowPack(
            text_to_front=default_text_to_front,
            zero123_orbits=default_zero123,
            guided_edit_front=default_guided,
            multiview_to_mesh=self.config.comfyui_multiview_to_mesh_workflow_path,
            image_to_mesh=self.config.comfyui_image_to_mesh_workflow_path,
            front_output="7",
            guided_front_output="21",
            orbit_outputs=orbit_defaults,
            mesh_output="17",
            image_mesh_output="11",
        )

    def _load_text_to_front_workflow(
        self, workflow_path: Path, *, prompt: str, run_id: str, seed: int | None = None
    ) -> dict[str, Any]:
        workflow = self._read_workflow(workflow_path)
        seed = seed if seed is not None else random.randint(1, 2**31 - 1)
        subject = self._normalize_subject_prompt(prompt)
        negative = self._build_view_negative(self.config.comfyui.negative_prompt)
        c = self.config.comfyui
        replacements: dict[str, Any] = {
            "__CHECKPOINT__": c.checkpoint,
            "__PROMPT__": self._build_view_prompt(subject, "front"),
            "__NEGATIVE_PROMPT__": negative,
            "__WIDTH__": c.width,
            "__HEIGHT__": c.height,
            "__STEPS__": c.steps,
            "__CFG__": c.cfg,
            "__SAMPLER__": c.view_sampler or "euler",
            "__SCHEDULER__": c.view_scheduler or "sgm_uniform",
            "__SEED__": seed,
            "__OUTPUT_PREFIX__": f"meshforge/{run_id}/front",
        }
        logger.info("text→front seed=%s subject=%s", seed, subject[:120])
        return self._render_workflow(workflow, replacements)

    def _load_zero123_orbits_workflow(
        self,
        workflow_path: Path,
        *,
        uploaded_front: str,
        run_id: str,
        seed: int | None = None,
    ) -> dict[str, Any]:
        workflow = self._read_workflow(workflow_path)
        seed = seed if seed is not None else random.randint(1, 2**31 - 1)
        c = self.config.comfyui
        replacements: dict[str, Any] = {
            "__ZERO123_CHECKPOINT__": c.zero123_checkpoint or "stable_zero123.ckpt",
            "__FRONT_IMAGE__": uploaded_front,
            "__ZERO123_WIDTH__": int(c.zero123_width),
            "__ZERO123_HEIGHT__": int(c.zero123_height),
            "__ZERO123_STEPS__": int(c.zero123_steps),
            "__ZERO123_CFG__": float(c.zero123_cfg),
            "__ZERO123_SAMPLER__": c.zero123_sampler or "euler",
            "__ZERO123_SCHEDULER__": c.zero123_scheduler or "normal",
            "__ZERO123_ELEVATION__": float(c.zero123_elevation),
            "__AZIMUTH_LEFT__": float(c.zero123_azimuth_left),
            "__AZIMUTH_BACK__": float(c.zero123_azimuth_back),
            "__AZIMUTH_RIGHT__": float(c.zero123_azimuth_right),
            "__SEED__": seed,
            "__OUTPUT_LEFT__": f"meshforge/{run_id}/left",
            "__OUTPUT_BACK__": f"meshforge/{run_id}/back",
            "__OUTPUT_RIGHT__": f"meshforge/{run_id}/right",
        }
        logger.info("zero123 orbits seed=%s front=%s", seed, uploaded_front)
        return self._render_workflow(workflow, replacements)

    def _load_multiview_to_mesh_workflow(
        self,
        workflow_path: Path,
        *,
        uploaded_views: dict[str, str],
        run_id: str,
        seed: int | None = None,
    ) -> dict[str, Any]:
        workflow = self._read_workflow(workflow_path)
        present = {key: value for key, value in uploaded_views.items() if key in VIEW_LABELS and value}
        if not present:
            raise RuntimeError("No views uploaded for multiview reconstruction")
        if "front" not in present:
            first = next(iter(present))
            present = {"front": present[first], **{k: v for k, v in present.items() if k != first}}
        node_map = {
            "front": ("3", "4"),
            "left": ("5", "6"),
            "back": ("7", "8"),
            "right": ("9", "10"),
        }
        cond_inputs = workflow.get("11", {}).get("inputs", {})
        for label, (load_id, enc_id) in node_map.items():
            if label not in present:
                workflow.pop(load_id, None)
                workflow.pop(enc_id, None)
                cond_inputs.pop(label, None)
        replacements: dict[str, Any] = {
            "__MESH_CHECKPOINT__": self.config.comfyui.mesh_checkpoint,
            "__MESH_RESOLUTION__": self.config.comfyui.mesh_resolution,
            "__MESH_STEPS__": self.config.comfyui.mesh_steps,
            "__MESH_CFG__": self.config.comfyui.mesh_cfg,
            "__MESH_GUIDANCE__": self.config.comfyui.mesh_guidance,
            "__MESH_OCTREE__": self.config.comfyui.mesh_octree_resolution,
            "__MESH_CHUNKS__": self.config.comfyui.mesh_num_chunks,
            "__MESH_SEED__": seed if seed is not None else random.randint(1, 2**31 - 1),
            "__MESH_OUTPUT__": f"meshforge/{run_id}/mesh",
        }
        placeholder = {
            "front": "__FRONT_IMAGE__",
            "left": "__LEFT_IMAGE__",
            "back": "__BACK_IMAGE__",
            "right": "__RIGHT_IMAGE__",
        }
        for label, uploaded in present.items():
            replacements[placeholder[label]] = uploaded
        return self._render_workflow(workflow, replacements)

    def _load_image_to_mesh_workflow(
        self,
        workflow_path: Path,
        *,
        uploaded_front: str,
        run_id: str,
        seed: int | None = None,
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
            "__MESH_SEED__": seed if seed is not None else random.randint(1, 2**31 - 1),
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
        assigned: dict[str, Path] = dict(labeled)
        for path in unlabeled:
            for label in VIEW_LABELS:
                if label not in assigned:
                    assigned[label] = path
                    break
        if not assigned and unlabeled:
            assigned["front"] = unlabeled[0]
        return assigned

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

    def _normalize_subject_prompt(self, prompt: str) -> str:
        from mesh_forge.backends.lmstudio import _trim_subject_prompt

        return _trim_subject_prompt(prompt)

    def _view_style(self) -> str:
        style = (self.config.comfyui.view_style or "clay").strip().lower()
        return style if style in {"clay", "color"} else "clay"

    def _build_view_negative(self, base_negative: str) -> str:
        extras = _VIEW_NEGATIVE_EXTRAS
        if self._view_style() == "clay":
            extras = f"{extras}, {_CLAY_NEGATIVE_EXTRAS}"
        base = (base_negative or "").strip().rstrip(",")
        return f"{base}, {extras}" if base else extras

    def _build_view_prompt(self, prompt: str, label: str) -> str:
        return isolation_view_prompt(
            self._normalize_subject_prompt(prompt),
            self._view_style(),
            view=label,
        )

    def _collect_front_image(
        self,
        client: httpx.Client,
        *,
        history: dict[str, Any],
        output_dir: Path,
        output_node: str,
    ) -> ImageSet:
        output_dir.mkdir(parents=True, exist_ok=True)
        outputs = history.get("outputs", {})
        node_data = outputs.get(str(output_node)) or {}
        record = self._first_output_record(node_data)
        if not record:
            raise RuntimeError("ComfyUI produced no front view image")
        suffix = Path(record["filename"]).suffix.lower() or ".png"
        dest = output_dir / f"front{suffix}"
        self._download_output(client, record, dest)
        return ImageSet(items=[ImageArtifact(path=dest, label="front", role="view", stage="views")])

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
        outputs = history.get("outputs") or {}
        node_data = outputs.get(str(output_node)) or {}
        record = self._first_output_record(node_data)
        if not record:
            record = self._first_mesh_record(outputs)
        if not record:
            failure = self._history_failure(history)
            if failure:
                raise RuntimeError(f"ComfyUI mesh workflow failed ({failure})")
            keys = ", ".join(str(k) for k in outputs) or "none"
            raise RuntimeError(
                f"ComfyUI produced no mesh artifact (node {output_node}; history nodes: {keys})"
            )
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

    def _as_output_record(self, item: Any) -> dict[str, Any] | None:
        if not isinstance(item, dict):
            return None
        name = str(item.get("filename") or item.get("name") or "").strip()
        if not name:
            return None
        record = dict(item)
        record.setdefault("filename", name)
        return record

    def _first_output_record(self, node_data: dict[str, Any]) -> dict[str, Any] | None:
        if not isinstance(node_data, dict):
            return None
        for values in node_data.values():
            if isinstance(values, list):
                for item in values:
                    record = self._as_output_record(item)
                    if record:
                        return record
            else:
                record = self._as_output_record(values)
                if record:
                    return record
        return None

    def _first_mesh_record(self, outputs: dict[str, Any]) -> dict[str, Any] | None:
        fallback: dict[str, Any] | None = None
        if not isinstance(outputs, dict):
            return None
        for node_data in outputs.values():
            record = self._first_output_record(node_data if isinstance(node_data, dict) else {})
            if not record:
                continue
            suffix = Path(str(record.get("filename") or "")).suffix.lower()
            if suffix in _MESH_SUFFIXES:
                return record
            if fallback is None:
                fallback = record
        return fallback

    def _history_failure(self, history: dict[str, Any]) -> str | None:
        status = history.get("status") if isinstance(history, dict) else None
        if not isinstance(status, dict):
            return None
        for item in status.get("messages") or []:
            if not isinstance(item, (list, tuple)) or not item:
                continue
            if str(item[0] or "") != "execution_error":
                continue
            payload = item[1] if len(item) > 1 and isinstance(item[1], dict) else {}
            node = str(payload.get("node_type") or payload.get("node_id") or "").strip()
            message = str(
                payload.get("exception_message") or payload.get("exception_type") or "execution error"
            ).strip().splitlines()[0][:400]
            return f"{node}: {message}" if node else message
        if str(status.get("status_str") or "").lower() == "error":
            return "workflow status=error"
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
