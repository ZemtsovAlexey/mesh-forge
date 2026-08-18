# MeshForge

Локальный чат-агент для 3D: пишет, вызывает тулы, показывает картинки и интерактивный mesh в ленте.

## Что умеет

- чат как в Cursor: тулы свёрнутыми карточками, стриминг, Stop
- картинки в ленте (lightbox), STL/OBJ — Three.js прямо в сообщении (на весь экран)
- агент на [pydantic-ai](https://ai.pydantic.dev/) + OpenAI-compatible LLM (LM Studio или облачный API)
- тулы: `generate_image`, `generate_views`, `remove_background`, `images_to_mesh` (1–4 фото), `look`, `mask_mesh`, `remove_mesh`, `restore_mesh`, `orient_mesh`, `scale_mesh`, `smooth_mesh`, `remesh_mesh`, `fill_mesh`, `split_mesh`, `join_mesh`, `match_mesh`, `extract_mesh`, `offset_mesh`, `add_mesh`, `restore_patch`
- knobs на каждый generate-вызов: seed, quality (draft/quality), steps, cfg, style, guidance

## Быстрый старт

```powershell
cd C:\Users\ZemtsovAlexey\Projects\mesh-forge
.\scripts\setup.ps1
.\scripts\start-comfyui.ps1
.\scripts\run.ps1
```

Или одной командой с автозапуском ComfyUI:

```powershell
.\scripts\run.ps1 -WithComfyUI
```

UI: `http://<host>:7860`

Нужен **LLM** с function calling и **ComfyUI** (`http://127.0.0.1:8188`). Vision-модель — отдельно в настройках чата.

LLM — любой OpenAI Chat Completions endpoint:

- **LM Studio** локально: `http://127.0.0.1:1234/v1`, ключ `lm-studio`
- **OpenAI-compatible** (например [AI Tunnel](https://aitunnel.ru)): `https://api.aitunnel.ru/v1`, ключ `sk-aitunnel-…`, модель вроде `gpt-5.6-luna`

Переключатель провайдера — в ⚙ Настройках. Облачный LLM не занимает локальный GPU и не выгружает ComfyUI.

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

ComfyUI можно держать на другой машине: в Настройках укажи URL API, например `http://192.168.1.20:8188`. На том ПК ComfyUI должен слушать не только localhost:

```powershell
.\scripts\start-comfyui.ps1 -ListenHost 0.0.0.0
```

Чекпоинты и custom nodes нужны на сервере ComfyUI. Если LLM и Comfy на разных хостах, GPU-очереди независимы (выгрузки VRAM нет).

## Конфиг

Первый запуск создаёт `config.yaml`. Проверь `llm.*`, `comfyui.*`, `paths.projects`. `gpu.sequential_models` выгружает LLM/Comfy при смене слота — только если оба на одном хосте (для 8GB VRAM) и LLM — локальный LM Studio. Если `llm.provider: openai` или `llm.base_url` и `comfyui.base_url` указывают на разные машины, очереди независимы и выгрузки нет. Принудительно: `gpu.shared_gpu: true|false`. Один и тот же ПК должен быть одним хостом в обоих URL (не `127.0.0.1` у LLM и LAN-IP у Comfy). ComfyUI стартует с `--disable-smart-memory`; перед локальным LM Studio агент ждёт падения VRAM после `/free` и при необходимости перезапускает локальный процесс.

## Скрипты

- `scripts/setup.ps1` — `uv sync` + ComfyUI + checkpoints
- `scripts/start-comfyui.ps1` / `stop-comfyui.ps1`
- `scripts/run.ps1` — FastAPI (`-WithComfyUI` опционально)

UI собирается сам: при старте и при открытии `/`, если `web/src` новее `web/dist`. Достаточно обновить страницу. Нужен `npm` в PATH.
