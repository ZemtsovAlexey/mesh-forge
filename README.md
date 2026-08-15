# MeshForge

Локальный сервер для подготовки печатаемых 3D-моделей из текста, набора фото и готового mesh в любой комбинации. Браузер — клиент, все сервисы работают на серверной машине.

## Что умеет сейчас

- единый pipeline `text / images / mesh -> STL`
- text-to-3D через ComfyUI end-to-end (`text → named views → mesh`)
- image-to-3D через ComfyUI (`1 фото` или `до 4 ракурсов`)
- cleanup и редактирование существующего mesh
- очередь GPU: один слот на LM Studio и ComfyUI; чат ждёт генерацию и наоборот. При `gpu.sequential_models: true` (по умолчанию) модели выгружаются при смене потребителя — нужно на 8GB VRAM. Позиция в очереди видна в прогрессе и в статус-пилюле `gpu`.

## Быстрый старт

Зависимости управляются через [uv](https://docs.astral.sh/uv/) (`pyproject.toml` + `uv.lock`).

```powershell
cd C:\AI\mesh-forge
.\scripts\setup.ps1
.\scripts\start-comfyui.ps1
.\scripts\run.ps1
```

Или одной командой с автозапуском ComfyUI:

```powershell
.\scripts\run.ps1 -WithComfyUI
```

Вручную:

```powershell
uv sync
uv run python .\server.py
```

UI:

```text
http://<server-ip>:7860
```

## Обязательные внешние сервисы

- `LM Studio` local server: `http://127.0.0.1:1234/v1`
- `ComfyUI` API: `http://127.0.0.1:8188`

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

Первый запуск создаёт `config.yaml` из example. Проверь:

- `paths.projects`
- `gpu.vram_gb` (NVIDIA определяется автоматически, для iGPU можно указать вручную)
- `gpu.sequential_models` (`true` = выгружать LLM/ComfyUI при смене слота; `false` = только FIFO, без unload)
- `llm.*`
- `comfyui.install_dir`
- `comfyui.*` checkpoints / workflow paths

`setup.ps1` идемпотентен: повторный запуск не перекачивает `uv`, portable ComfyUI или checkpoint'ы, если валидная установка уже есть.

## Локальные скрипты

- `scripts/setup.ps1` — `uv sync` + ComfyUI + checkpoints
- `scripts/setup-comfyui.ps1` — только ComfyUI
- `scripts/start-comfyui.ps1` / `scripts/stop-comfyui.ps1`
- `scripts/run.ps1` — FastAPI через `uv run` (`-WithComfyUI` опционально)

## Mesh Post-Processing

Blender больше не входит в runtime. Если вы задаёте `solidify_mm`, MeshForge оставит STL как есть и подскажет перенести толщину стенки в слайсер.
