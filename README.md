# MeshForge

Локальный сервер для подготовки печатаемых 3D-моделей из текста, набора фото и готового mesh в любой комбинации. Браузер — клиент, все сервисы работают на серверной машине.

## Что умеет сейчас

- единый pipeline `text / images / mesh -> STL`
- text-to-3D через ComfyUI end-to-end (`text → named views → mesh`)
- image-to-3D через ComfyUI (`1 фото` или `до 4 ракурсов`)
- cleanup и редактирование существующего mesh
- очередь GPU: тяжёлые задачи выполняются последовательно на локальной машине

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

`setup.ps1` ставит официальный **Windows portable** ComfyUI и сам выбирает архив по GPU:

- `ComfyUI_windows_portable_nvidia.7z`
- `ComfyUI_windows_portable_amd.7z`
- `ComfyUI_windows_portable_intel.7z`

По умолчанию portable ставится в `C:\AI\ComfyUI_windows_portable`. В `config.yaml` путь к приложению:

`comfyui.install_dir: C:/AI/ComfyUI_windows_portable/ComfyUI`

Checkpoint’ы:

- Draft (по умолчанию): `sd_xl_turbo` + `hunyuan3d-dit-v2-mv-turbo`
- Quality: `sd_xl_base_1.0_0.9vae` + `hunyuan3d-dit-v2-mv` (без turbo) — чище mesh, дольше

Переключатель в UI: **⚙ Генерация**. Скачать quality-модели:

`.\scripts\setup-comfyui.ps1 -QualityModels`

Процессы:

- `.\scripts\start-comfyui.ps1` — поднять API (`python_embeded`) и дождаться `/system_stats`
- `.\scripts\stop-comfyui.ps1` — остановить tracked/listening процесс
- pid/log: `.runtime/comfyui.pid`, `.runtime/comfyui.out.log`, `.runtime/comfyui.err.log`

## Конфиг

Первый запуск создаёт `config.yaml` из example. Проверь:

- `paths.projects`
- `gpu.vram_gb` (NVIDIA определяется автоматически, для iGPU можно указать вручную)
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
