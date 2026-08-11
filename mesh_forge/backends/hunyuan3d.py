from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

from mesh_forge.config import load_config
from mesh_forge import progress as prog

logger = logging.getLogger("mesh_forge.hunyuan3d")

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

_STAGE_RULES: list[tuple[re.Pattern[str], float, str]] = [
    (re.compile(r"STAGE:\s*loading_model", re.I), 22, "Загрузка Hunyuan3D-2mini…"),
    (re.compile(r"STAGE:\s*weights_cached", re.I), 28, "Веса уже в кэше…"),
    (re.compile(r"STAGE:\s*weights_linked|STAGE:\s*weights_copied", re.I), 30, "Подготовка локальных весов…"),
    (re.compile(r"STAGE:\s*resolve_weights", re.I), 26, "Проверка весов Hugging Face…"),
    (re.compile(r"STAGE:\s*model_ready", re.I), 38, "Модель загружена"),
    (re.compile(r"STAGE:\s*load_image", re.I), 42, "Чтение изображения…"),
    (re.compile(r"STAGE:\s*rembg\b", re.I), 46, "Удаление фона…"),
    (re.compile(r"STAGE:\s*rembg_done", re.I), 52, "Фон убран"),
    (re.compile(r"STAGE:\s*resize\b", re.I), 44, "Уменьшение изображения…"),
    (re.compile(r"STAGE:\s*inference\b", re.I), 58, "Инференс Hunyuan…"),
    (re.compile(r"STAGE:\s*set_flashvdm", re.I), 60, "Переключение декодера…"),
    (re.compile(r"STAGE:\s*inference_ok", re.I), 76, "Меш получен"),
    (re.compile(r"STAGE:\s*inference_done", re.I), 78, "Инференс завершён"),
    (re.compile(r"STAGE:\s*export\b", re.I), 82, "Экспорт OBJ…"),
    (re.compile(r"STAGE:\s*export_done", re.I), 86, "Экспорт готов"),
    (re.compile(r"try to download from huggingface", re.I), 24, "Проверка весов Hugging Face…"),
    (re.compile(r"Downloading data from .*u2net", re.I), 48, "Скачивание модели rembg (один раз)…"),
]
_TQDM_PCT = re.compile(r"(\d{1,3})%\|")
_FETCHING = re.compile(r"Fetching\s+(\d+)\s+files", re.I)
_BYTES_TQDM = re.compile(
    r"(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)\s*(GB|MB|KB)",
    re.I,
)


def _docker_path(path: Path) -> str:
    return path.resolve().as_posix()


def _docker_available() -> bool:
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=30,
            encoding="utf-8",
            errors="replace",
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _docker_image_ready(image: str) -> bool:
    result = subprocess.run(
        ["docker", "image", "inspect", image],
        capture_output=True,
        text=True,
        timeout=30,
        encoding="utf-8",
        errors="replace",
    )
    return result.returncode == 0


def _stage_input_image(image_path: Path, work_dir: Path) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    suffix = image_path.suffix.lower() or ".png"
    staged = work_dir / f"input{suffix}"
    if image_path.resolve() != staged.resolve():
        shutil.copy2(image_path, staged)
    return staged


def _parse_progress(line: str, project_id: str) -> None:
    text = line.strip()
    if not text:
        return
    # HF hub "Fetching N files" often means cache resolve (instant 100%), not a real download
    m_fetch = _FETCHING.search(text)
    if m_fetch:
        m_pct = re.search(r"(\d{1,3})%", text)
        done = int(m_pct.group(1)) if m_pct else 0
        total = max(1, int(m_fetch.group(1)))
        m_bytes = _BYTES_TQDM.search(text)
        if m_bytes:
            cur, tot, unit = float(m_bytes.group(1)), float(m_bytes.group(2)), m_bytes.group(3).upper()
            mapped = 24 + 12 * min(1.0, cur / max(tot, 1e-6))
            prog.update(project_id, mapped, f"Скачивание модели… {cur:.1f}/{tot:.1f} {unit}")
            return
        if done >= 100 or "it/s" in text:
            prog.update(project_id, 30, "Веса из кэша Hugging Face…")
            return
        mapped = 24 + 12 * (done / 100.0)
        label = (
            f"Проверка весов Hugging Face… {done}% ({total} файлов)"
            if done > 0
            else f"Проверка / скачивание весов… ({total} файлов)"
        )
        prog.update(project_id, mapped if done > 0 else 25, label)
        return
    for pattern, percent, stage in _STAGE_RULES:
        if pattern.search(text):
            prog.update(project_id, percent, stage)
            return
    m = _TQDM_PCT.search(text)
    if m:
        pct = min(99, int(m.group(1)))
        mapped = 22 + (pct / 100.0) * 14
        prog.update(project_id, mapped, f"Скачивание модели… {pct}%")


def _run_subprocess(
    cmd: list[str],
    *,
    label: str,
    project_id: str | None = None,
    timeout_s: int = 1800,
) -> None:
    logger.info("%s cmd: %s", label, " ".join(cmd))
    started = time.perf_counter()
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        bufsize=1,
    )
    assert proc.stdout is not None
    tail: list[str] = []

    def _reader() -> None:
        for raw in proc.stdout:
            line = raw.rstrip("\n")
            tail.append(line)
            if len(tail) > 400:
                del tail[:200]
            if project_id:
                _parse_progress(line, project_id)
            low = line.lower()
            if any(k in low for k in ("stage:", "error", "traceback", "cuda", "oom", "download")):
                logger.info("%s | %s", label, line[:240])

    reader = threading.Thread(target=_reader, daemon=True)
    reader.start()

    try:
        rc = proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        reader.join(timeout=2)
        raise RuntimeError(f"{label} timed out after {timeout_s}s") from None

    reader.join(timeout=5)
    elapsed = time.perf_counter() - started
    log_blob = "\n".join(tail[-80:])
    if rc != 0:
        logger.error("%s failed rc=%s after %.1fs\n%s", label, rc, elapsed, log_blob[-4000:])
        raise RuntimeError(f"{label} failed:\n{log_blob[-3000:]}")
    logger.info("%s done in %.1fs", label, elapsed)


def _infer_args(*, remove_bg: bool) -> list[str]:
    cfg = load_config()
    args = [
        "--model-path",
        cfg.docker.hunyuan_model,
        "--subfolder",
        cfg.docker.hunyuan_subfolder,
        "--steps",
        str(cfg.docker.hunyuan_steps),
        "--octree-resolution",
        str(cfg.docker.hunyuan_octree),
        "--num-chunks",
        str(cfg.docker.hunyuan_chunks),
    ]
    if not remove_bg:
        args.append("--no-remove-bg")
    if cfg.gpu.vram_gb <= 6:
        # Safer defaults on tiny cards
        args.extend(["--octree-resolution", "192", "--num-chunks", "4000", "--steps", "16"])
    return args


def run_hunyuan3d(
    image_path: Path,
    output_dir: Path,
    *,
    remove_bg: bool = True,
    project_id: str | None = None,
) -> Path:
    cfg = load_config()
    if not cfg.docker.enabled:
        raise RuntimeError(
            "Hunyuan3D requires Docker. Set docker.enabled: true and build "
            "docker/hunyuan3d/build.ps1"
        )
    if not _docker_available():
        raise RuntimeError(
            "Docker is not available. Install Docker Desktop or build/run Hunyuan later."
        )
    image = cfg.docker.hunyuan_image
    if not _docker_image_ready(image):
        raise RuntimeError(
            f"Docker image not found: {image}. Build with: .\\docker\\hunyuan3d\\build.ps1"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = output_dir.parent
    if project_id:
        prog.update(project_id, 12, "Подготовка входа…")
    staged = _stage_input_image(image_path, work_dir)
    hf_cache = cfg.hf_cache_dir
    hf_cache.mkdir(parents=True, exist_ok=True)
    hy3d_cache = hf_cache.parent / "hy3dgen"
    u2net_cache = hf_cache.parent / "u2net"
    hy3d_cache.mkdir(parents=True, exist_ok=True)
    u2net_cache.mkdir(parents=True, exist_ok=True)
    infer_py = _PROJECT_ROOT / "docker" / "hunyuan3d" / "infer.py"

    cmd = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "-e",
        "PYTHONUNBUFFERED=1",
        # HF XET backend can stall for minutes at "Fetching N files: 0%"
        "-e",
        "HF_HUB_DISABLE_XET=1",
        "-e",
        "HY3DGEN_MODELS=/root/.cache/hy3dgen",
        "-e",
        "U2NET_HOME=/root/.u2net",
        "-v",
        f"{_docker_path(work_dir)}:/work",
        "-v",
        f"{_docker_path(hf_cache)}:/root/.cache/huggingface",
        "-v",
        f"{_docker_path(hy3d_cache)}:/root/.cache/hy3dgen",
        "-v",
        f"{_docker_path(u2net_cache)}:/root/.u2net",
    ]
    # Always use repo infer.py so cache/rembg fixes apply without image rebuild
    if infer_py.is_file():
        cmd.extend(["-v", f"{_docker_path(infer_py)}:/opt/hunyuan/infer.py:ro"])
    cmd.extend(
        [
            image,
            f"/work/{staged.name}",
            "--output-dir",
            f"/work/{output_dir.name}",
            *_infer_args(remove_bg=remove_bg),
        ]
    )
    if project_id:
        prog.update(project_id, 18, "Docker Hunyuan3D…")
    logger.info(
        "Hunyuan3D start image=%s output=%s docker=%s",
        image_path,
        output_dir,
        image,
    )
    _run_subprocess(cmd, label="Hunyuan3D/docker", project_id=project_id)

    if project_id:
        prog.update(project_id, 86, "Сбор результатов…")
    objs = list(output_dir.glob("**/*.obj"))
    if not objs:
        raise RuntimeError("Hunyuan3D produced no OBJ output")
    logger.info("Hunyuan3D output: %s", objs[0])
    return objs[0]


def hunyuan3d_available() -> bool:
    cfg = load_config()
    if not cfg.docker.enabled:
        return False
    return _docker_available() and _docker_image_ready(cfg.docker.hunyuan_image)
