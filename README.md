# MeshForge

Локальный сервер для подготовки печатаемых 3D-моделей из текста, набора фото и готового mesh в любой комбинации. Браузер — клиент, все сервисы работают на серверной машине.

## Что умеет сейчас

- единый pipeline `text / images / mesh -> STL`
- text-to-3D через ComfyUI end-to-end (`text → named views → mesh`)
- image-to-3D через ComfyUI (`1 фото` или `до 4 ракурсов`)
- cleanup и редактирование существующего mesh
- очередь GPU: тяжёлые задачи выполняются последовательно под 8 GB VRAM

## Быстрый старт

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

UI:

```text
http://<server-ip>:7860
```

## Обязательные внешние сервисы

- `LM Studio` local server: `http://127.0.0.1:1234/v1`
- `ComfyUI` API: `http://127.0.0.1:8188`
- `Blender` для solidify / repair

## ComfyUI

`setup.ps1` ставит ComfyUI в `C:\AI\ComfyUI` (или `comfyui.install_dir` из config) и скачивает checkpoint’ы:

- `sd_xl_turbo_1.0_fp16.safetensors` — text→views
- `hunyuan3d-dit-v2-mv-turbo_fp16.safetensors` — photo/text multiview→mesh (одно фото: недостающие ракурсы = front)

Процессы:

- `.\scripts\start-comfyui.ps1` — поднять API и дождаться `/system_stats`
- `.\scripts\stop-comfyui.ps1` — остановить tracked/listening процесс
- pid/log: `.runtime/comfyui.pid`, `.runtime/comfyui.out.log`, `.runtime/comfyui.err.log`

## Конфиг

Первый запуск создаёт `config.yaml` из example. Проверь:

- `paths.blender`
- `paths.projects`
- `llm.*`
- `comfyui.install_dir`
- `comfyui.*` checkpoints / workflow paths

## Локальные скрипты

- `scripts/setup.ps1` — venv MeshForge + ComfyUI + checkpoints
- `scripts/setup-comfyui.ps1` — только ComfyUI
- `scripts/start-comfyui.ps1` / `scripts/stop-comfyui.ps1`
- `scripts/run.ps1` — FastAPI сервер (`-WithComfyUI` опционально)
