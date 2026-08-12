# MeshForge — план проекта

Локальный пайплайн «фото / скан / текст → 3D-модель → 3D-печать» с веб-UI.

## Архитектура

- **Клиент:** браузер (с любой машины в LAN)
- **Сервер (DESKTOP-HOME):** LM Studio, ComfyUI, MeshForge, Blender headless
- **LLM:** LM Studio (OpenAI-compatible API) на сервере

## Три ветки создания

| Ветка | Вход | Инструмент |
|-------|------|------------|
| 1. Фото | JPG/PNG | ComfyUI Hunyuan MV → Blender QC |
| 2. Скан | STL/OBJ | PyMeshLab / Poisson → Blender |
| 3. Текст | промпт | ComfyUI text→views→mesh |

## Редактирование

- Text edit: LLM planner → JSON ops → Blender/trimesh
- Photo edit: VLM diff → image-to-3D → merge
- Версионирование: manifest.yaml (v1, v2, v3...)

---

## Фазы разработки

### Фаза 0 — Инфраструктура ✅ (частично)
- [x] Локальные скрипты (`scripts/setup.ps1`, `start-comfyui.ps1`, `run.ps1`)
- [x] Установка серверного стека (Python, ComfyUI)
- [ ] LM Studio + модели
- [x] MeshForge FastAPI + UI

### Фаза 1 — UI каркас
- [ ] Gradio: проекты, создать, редактировать, история, экспорт
- [ ] manifest + mesh_qc

### Фаза 2 — Branch 2: Скан
- [ ] PyMeshLab pipeline
- [ ] Blender watertight QC

### Фаза 3 — Branch 1: Фото
- [ ] ComfyUI image→mesh
- [ ] rembg

### Фаза 4 — Branch 3: Текст
- [ ] ComfyUI text→views→mesh
- [ ] OpenSCAD + LM Studio (опционально)

### Фаза 5 — Редактирование
- [ ] LLM planner (LM Studio API)
- [ ] Text/photo edit executors

### Фаза 6 — Полировка
- [ ] Task Scheduler автозапуск
- [ ] Документация

---

## Локальные скрипты (`scripts/`)

| Скрипт | Назначение |
|--------|------------|
| `setup.ps1` | venv MeshForge + ComfyUI + checkpoints |
| `setup-comfyui.ps1` | только ComfyUI |
| `start-comfyui.ps1` / `stop-comfyui.ps1` | управление ComfyUI |
| `run.ps1` | FastAPI (`-WithComfyUI` опционально) |

```powershell
.\scripts\setup.ps1
.\scripts\start-comfyui.ps1
.\scripts\run.ps1
```

---

## Железо

| | DESKTOP-HOME |
|--|--------------|
| GPU | RTX 3070 Ti 8GB |
| RAM | 32 GB |
| Роль | Compute + LM Studio + MeshForge |

## Документация

- [README.md](README.md) — быстрый старт
- [PLAN.md](PLAN.md) — этот файл
