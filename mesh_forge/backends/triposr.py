from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from mesh_forge.config import load_config
from mesh_forge import progress as prog

logger = logging.getLogger("mesh_forge.triposr")

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_TRIPOSR_VENV_PYTHON = _PROJECT_ROOT / "venv-triposr" / "Scripts" / "python.exe"
_VENDOR_DIR = Path(__file__).resolve().parent.parent / "vendor"
_LAUNCHER = Path(__file__).resolve().parent / "_triposr_launcher.py"

# TripoSR run.py stages → (percent, UI label)
_STAGE_RULES: list[tuple[re.Pattern[str], float, str]] = [
    (re.compile(r"Initializing model \.\.\.", re.I), 22, "Загрузка модели TripoSR…"),
    (re.compile(r"Initializing model finished", re.I), 36, "Модель загружена"),
    (re.compile(r"Processing images \.\.\.", re.I), 40, "Удаление фона / подготовка…"),
    (re.compile(r"Processing images finished", re.I), 55, "Изображение готово"),
    (re.compile(r"Running image\s+(\d+)/(\d+)", re.I), 58, "Инференс…"),
    (re.compile(r"Running model \.\.\.", re.I), 60, "Инференс нейросети…"),
    (re.compile(r"Running model finished", re.I), 72, "Инференс завершён"),
    (re.compile(r"Extracting mesh finished", re.I), 82, "Меш извлечён"),
    (re.compile(r"Extracting mesh \.\.\.", re.I), 75, "Извлечение меша…"),
    (re.compile(r"Exporting mesh.*?finished", re.I), 86, "Экспорт готов"),
    (re.compile(r"Exporting mesh", re.I), 84, "Экспорт OBJ…"),
]
_TQDM_PCT = re.compile(r"(\d{1,3})%\|")
_TQDM_BYTES = re.compile(r"(\d+(?:\.\d+)?[kKmMgG]?)/(\d+(?:\.\d+)?[kKmMgG]?)")


def triposr_path() -> Path:
    cfg = load_config()
    if cfg.paths.triposr:
        return Path(cfg.paths.triposr)
    raise FileNotFoundError("TripoSR path not set in config.yaml")


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
        logger.debug("Staged input %s -> %s", image_path, staged)
    return staged


def _triposr_args(*, remove_bg: bool) -> list[str]:
    args: list[str] = ["--model-save-format", "obj"]
    if not remove_bg:
        args.append("--no-remove-bg")
    cfg = load_config()
    # Explicit resolution: default TripoSR is 256; keep it for 8GB, lower only on tiny VRAM.
    if cfg.gpu.vram_gb <= 6:
        args.extend(["--chunk-size", "200", "--mc-resolution", "128"])
    else:
        # 8GB+: prefer denser isosurface (default TripoSR is 256)
        args.extend(["--mc-resolution", "320"])
    return args


def _parse_triposr_progress(line: str, project_id: str) -> None:
    text = line.strip()
    if not text:
        return
    for pattern, percent, stage in _STAGE_RULES:
        m = pattern.search(text)
        if not m:
            continue
        if pattern.pattern.startswith(r"Running image"):
            cur, total = int(m.group(1)), max(1, int(m.group(2)))
            percent = 58 + 12 * ((cur - 1) / total)
            stage = f"Инференс {cur}/{total}…"
        prog.update(project_id, percent, stage)
        return
    # HuggingFace / rembg download bars (first run)
    m = _TQDM_PCT.search(text)
    if m:
        pct = min(99, int(m.group(1)))
        # Map download 0-100% into model-load window 22-35
        mapped = 22 + (pct / 100.0) * 13
        prog.update(project_id, mapped, f"Скачивание модели… {pct}%")


def _run_subprocess(
    cmd: list[str],
    *,
    cwd: str | None,
    env: dict[str, str] | None,
    label: str,
    project_id: str | None = None,
) -> None:
    logger.info("%s cmd: %s", label, " ".join(cmd))
    started = time.perf_counter()
    run_env = dict(env or os.environ)
    run_env.setdefault("PYTHONUNBUFFERED", "1")
    # Force line-buffered logs from Python inside the container/process
    run_env.setdefault("PYTHONIOENCODING", "utf-8")

    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=run_env,
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
                _parse_triposr_progress(line, project_id)
            # Keep a light breadcrumb in app logs for debugging
            low = line.lower()
            if any(k in low for k in ("initializing", "processing", "running model", "extracting", "exporting", "error", "traceback")):
                logger.info("%s | %s", label, line[:240])

    reader = threading.Thread(target=_reader, daemon=True)
    reader.start()

    try:
        rc = proc.wait(timeout=900)
    except subprocess.TimeoutExpired:
        proc.kill()
        reader.join(timeout=2)
        raise RuntimeError(f"{label} timed out after 900s") from None

    reader.join(timeout=5)
    elapsed = time.perf_counter() - started
    log_blob = "\n".join(tail[-80:])
    if rc != 0:
        logger.error("%s failed rc=%s after %.1fs\n%s", label, rc, elapsed, log_blob[-4000:])
        raise RuntimeError(f"{label} failed:\n{log_blob[-3000:]}")
    logger.info("%s done in %.1fs", label, elapsed)


def _run_triposr_docker(
    image_path: Path,
    output_dir: Path,
    *,
    remove_bg: bool,
    project_id: str | None = None,
) -> None:
    cfg = load_config()
    work_dir = output_dir.parent
    if project_id:
        prog.update(project_id, 12, "Подготовка входа…")
    staged = _stage_input_image(image_path, work_dir)
    hf_cache = cfg.hf_cache_dir
    hf_cache.mkdir(parents=True, exist_ok=True)

    cmd = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "-e",
        "PYTHONUNBUFFERED=1",
        "-v",
        f"{_docker_path(work_dir)}:/work",
        "-v",
        f"{_docker_path(hf_cache)}:/root/.cache/huggingface",
        "--entrypoint",
        "python",
        cfg.docker.triposr_image,
        "-u",
        "run.py",
        f"/work/{staged.name}",
        "--output-dir",
        f"/work/{output_dir.name}",
        *_triposr_args(remove_bg=remove_bg),
    ]
    if project_id:
        prog.update(project_id, 18, "Docker TripoSR…")
    _run_subprocess(
        cmd,
        cwd=None,
        env=os.environ.copy(),
        label="TripoSR/docker",
        project_id=project_id,
    )


def _triposr_env() -> dict[str, str]:
    env = os.environ.copy()
    prefix = str(_VENDOR_DIR)
    env["PYTHONPATH"] = prefix + os.pathsep + env.get("PYTHONPATH", "")
    return env


def _triposr_python() -> str:
    if _TRIPOSR_VENV_PYTHON.is_file():
        return str(_TRIPOSR_VENV_PYTHON)
    return sys.executable


def _run_triposr_local(
    image_path: Path,
    output_dir: Path,
    *,
    remove_bg: bool,
    project_id: str | None = None,
) -> None:
    from mesh_forge.backends.triposr_shim import ensure_torchmcubes_shim

    ensure_torchmcubes_shim()
    triposr = triposr_path()
    run_py = triposr / "run.py"
    if not run_py.is_file():
        raise FileNotFoundError(f"TripoSR run.py not found: {run_py}")

    cmd = [
        _triposr_python(),
        str(_LAUNCHER),
        str(triposr),
        str(run_py),
        str(image_path),
        "--output-dir",
        str(output_dir),
        *_triposr_args(remove_bg=remove_bg),
    ]
    _run_subprocess(
        cmd,
        cwd=str(triposr),
        env=_triposr_env(),
        label="TripoSR/local",
        project_id=project_id,
    )


def run_triposr(
    image_path: Path,
    output_dir: Path,
    *,
    remove_bg: bool = True,
    project_id: str | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config()

    if cfg.docker.enabled:
        if not _docker_available():
            raise RuntimeError(
                "Docker is not available. Install Docker Desktop or set docker.enabled: false in config.yaml"
            )
        if not _docker_image_ready(cfg.docker.triposr_image):
            raise RuntimeError(
                f"Docker image not found: {cfg.docker.triposr_image}. "
                f"Build with: .\\docker\\triposr\\build.ps1"
            )
        logger.info(
            "TripoSR start (docker) image=%s output=%s docker_image=%s",
            image_path,
            output_dir,
            cfg.docker.triposr_image,
        )
        _run_triposr_docker(image_path, output_dir, remove_bg=remove_bg, project_id=project_id)
    else:
        logger.info("TripoSR start (local) image=%s output=%s", image_path, output_dir)
        _run_triposr_local(image_path, output_dir, remove_bg=remove_bg, project_id=project_id)

    if project_id:
        prog.update(project_id, 86, "Сбор результатов…")
    objs = list(output_dir.glob("**/*.obj"))
    if not objs:
        raise RuntimeError("TripoSR produced no OBJ output")
    logger.info("TripoSR output: %s", objs[0])
    return objs[0]


def triposr_available() -> bool:
    cfg = load_config()
    if cfg.docker.enabled:
        return _docker_available() and _docker_image_ready(cfg.docker.triposr_image)
    try:
        return (triposr_path() / "run.py").is_file()
    except FileNotFoundError:
        return False
