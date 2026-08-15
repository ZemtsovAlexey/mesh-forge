from __future__ import annotations

import logging
import os
import shutil
import subprocess
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse

logger = logging.getLogger("api.frontend")

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
DIST_DIR = WEB_DIR / "dist"

_BUILD_LOCK = threading.Lock()
_SOURCE_NAMES = (
    "index.html",
    "package.json",
    "package-lock.json",
    "vite.config.ts",
    "tsconfig.json",
    "tsconfig.node.json",
)
_INDEX_HEADERS = {
    "Cache-Control": "no-cache, no-store, must-revalidate",
    "Pragma": "no-cache",
}


def _npm() -> str | None:
    return shutil.which("npm") or shutil.which("npm.cmd")


def _source_mtime() -> float:
    latest = 0.0
    src = WEB_DIR / "src"
    if src.is_dir():
        for path in src.rglob("*"):
            if path.is_file():
                latest = max(latest, path.stat().st_mtime)
    for name in _SOURCE_NAMES:
        path = WEB_DIR / name
        if path.is_file():
            latest = max(latest, path.stat().st_mtime)
    return latest


def ui_is_stale() -> bool:
    index = DIST_DIR / "index.html"
    if not index.is_file():
        return True
    assets = DIST_DIR / "assets"
    if not assets.is_dir() or not any(assets.iterdir()):
        return True
    return _source_mtime() > index.stat().st_mtime + 0.5


def ensure_ui_built() -> None:
    if not ui_is_stale():
        return
    with _BUILD_LOCK:
        if not ui_is_stale():
            return
        npm = _npm()
        if not npm:
            raise RuntimeError("npm not found; cannot rebuild chat UI")
        if not (WEB_DIR / "node_modules").is_dir():
            logger.info("Installing web dependencies...")
            _run_npm(npm, "install")
        logger.info("Rebuilding chat UI (web/src newer than web/dist)...")
        _run_npm(npm, "run", "build")
        logger.info("Chat UI rebuilt")


def _run_npm(npm: str, *args: str) -> None:
    kwargs: dict = {
        "cwd": WEB_DIR,
        "check": False,
        "capture_output": True,
        "text": True,
    }
    if os.name == "nt":
        completed = subprocess.run(" ".join([f'"{npm}"', *args]), shell=True, **kwargs)
    else:
        completed = subprocess.run([npm, *args], **kwargs)
    if completed.returncode != 0:
        output = (completed.stderr or completed.stdout or "").strip()
        raise RuntimeError(output or f"npm {' '.join(args)} failed")


def attach_frontend(app: FastAPI) -> None:
    try:
        ensure_ui_built()
    except Exception:
        logger.exception("Initial UI build failed")

    @app.get("/")
    def index():
        try:
            ensure_ui_built()
        except Exception as exc:
            logger.exception("UI rebuild failed")
            index_file = DIST_DIR / "index.html"
            if not index_file.is_file():
                return JSONResponse(
                    {"detail": f"UI not built: {exc}"},
                    status_code=503,
                )
        index_file = DIST_DIR / "index.html"
        if index_file.is_file():
            return FileResponse(index_file, headers=_INDEX_HEADERS)
        return JSONResponse({"detail": "UI not built. Install npm and restart."}, status_code=503)

    @app.get("/assets/{path:path}")
    def asset(path: str):
        root = (DIST_DIR / "assets").resolve()
        file = (root / path).resolve()
        try:
            file.relative_to(root)
        except ValueError as exc:
            raise HTTPException(404, "Asset not found") from exc
        if not file.is_file():
            raise HTTPException(404, "Asset not found")
        return FileResponse(file)
