# MeshForge

Локальный чат-агент для 3D: пишет, вызывает тулы, показывает картинки и интерактивный mesh в ленте.

## Что умеет

- чат как в Cursor: тулы свёрнутыми карточками, стриминг, Stop
- картинки в ленте (lightbox), STL/OBJ — Three.js прямо в сообщении (на весь экран)
- агент на [pydantic-ai](https://ai.pydantic.dev/) + LM Studio
- тулы: `generate_image`, `generate_views`, `images_to_mesh` (1–4 фото, без pad до 4), `look`, `inspect_mesh`, `repair_mesh`, `orient_mesh`, `scale_mesh`, `smooth_mesh`, `decimate_mesh`
- knobs на каждый generate-вызов: seed, quality (draft/quality), steps, cfg, style, denoise, guidance

## Быстрый старт

```powershell
cd C:\Users\ZemtsovAlexey\Projects\mesh-forge
.\scripts\setup.ps1
cd web
npm install
npm run build
cd ..
.\scripts\start-comfyui.ps1
.\scripts\run.ps1
```

Или одной командой с автозапуском ComfyUI:

```powershell
.\scripts\run.ps1 -WithComfyUI
```

UI: `http://<host>:7860`

Нужны **LM Studio** (`http://127.0.0.1:1234/v1`) с моделью, у которой есть function calling, и **ComfyUI** (`http://127.0.0.1:8188`). Vision-модель — отдельно в настройках чата.

## ComfyUI

`setup-comfyui.ps1` сначала ищет уже установленный **ComfyUI Desktop**, иначе ставит официальный **Windows portable** (архив по GPU):

- Desktop: `%LOCALAPPDATA%\Programs\ComfyUI`, user data обычно `%USERPROFILE%\Documents\ComfyUI`
- Portable: `%LOCALAPPDATA%\MeshForge\ComfyUI_windows_portable` (`nvidia` / `amd` / `intel`)

В `config.yaml` пишется путь к **user data / models** (не к Electron-exe):

- Desktop: `comfyui.install_dir: C:/Users/<you>/Documents/ComfyUI`
- Portable: `.../ComfyUI_windows_portable/ComfyUI`

Пустой `install_dir` больше не означает «магический `C:\AI\...`» — скрипты делают discovery.
Каталог checkpoints берётся из живого ComfyUI (`GET /experiment/models`), если API доступен; иначе — `{install_dir|Desktop basePath}/models/checkpoints`.

Checkpoint’ы:

- Draft (по умолчанию): `sd_xl_turbo` + `hunyuan3d-dit-v2-mv-turbo`
- Quality: `sd_xl_base_1.0_0.9vae` + `hunyuan3d-dit-v2-mv` (без turbo) — чище mesh, дольше

Переключатель в UI: **⚙ Генерация**. Скачать quality-модели:

`.\scripts\setup-comfyui.ps1 -QualityModels`

Принудительно portable (игнорируя Desktop): `-ForcePortable`.

Процессы:

- `.\scripts\start-comfyui.ps1` — поднять API на `0.0.0.0:8188` (Desktop `.venv` + `--base-directory`, либо portable `python_embeded`) и дождаться `/system_stats`. Локально: `http://127.0.0.1:8188`; с LAN: `http://<server-ip>:8188`. Только localhost: `-ListenHost 127.0.0.1`
- `.\scripts\stop-comfyui.ps1` — остановить tracked/listening процесс
- pid/log: `.runtime/comfyui.pid`, `.runtime/comfyui.out.log`, `.runtime/comfyui.err.log`

## Конфиг

Первый запуск создаёт `config.yaml`. Проверь `llm.*`, `comfyui.*`, `paths.projects`, `gpu.sequential_models` (по умолчанию выгружать LLM/Comfy при смене слота — для 8GB VRAM).

## Скрипты

- `scripts/setup.ps1` — `uv sync` + ComfyUI + checkpoints
- `scripts/start-comfyui.ps1` / `stop-comfyui.ps1`
- `scripts/run.ps1` — FastAPI (`-WithComfyUI` опционально)

Frontend в dev: `cd web && npm run dev` (прокси на `:7860`). Для сервера нужна сборка `npm run build` → `web/dist`.
