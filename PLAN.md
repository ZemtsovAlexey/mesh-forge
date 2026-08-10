# MeshForge — план проекта

Локальный пайплайн «фото / скан / текст → 3D-модель → 3D-печать» с веб-UI.

## Архитектура

- **Клиент (Z13):** браузер, Blender, слайсер
- **Сервер (DESKTOP-HOME):** LM Studio, TripoSR, PyMeshLab, Blender headless
- **LLM:** LM Studio (OpenAI-compatible API), удалённо на сервере

## Три ветки создания

| Ветка | Вход | Инструмент |
|-------|------|------------|
| 1. Фото | JPG/PNG | TripoSR → Blender QC |
| 2. Скан | STL/OBJ | PyMeshLab / Poisson → Blender |
| 3. Текст | промпт | OpenSCAD+LLM / TRELLIS |

## Редактирование

- Text edit: LLM planner → JSON ops → Blender/trimesh
- Photo edit: VLM diff → image-to-3D → merge
- Версионирование: manifest.yaml (v1, v2, v3...)

---

## Фазы разработки

### Фаза 0 — Инфраструктура ✅ (частично)
- [x] SSH-доступ Z13 → DESKTOP-HOME
- [x] Скрипты развёртывания (`deploy/`)
- [x] Установка серверного стека (Python, PyTorch, TripoSR)
- [ ] LM Studio + модели
- [ ] MeshForge app.py + UI

### Фаза 1 — UI каркас
- [ ] Gradio: проекты, создать, редактировать, история, экспорт
- [ ] manifest + mesh_qc

### Фаза 2 — Branch 2: Скан
- [ ] PyMeshLab pipeline
- [ ] Blender watertight QC

### Фаза 3 — Branch 1: Фото
- [ ] TripoSR adapter
- [ ] rembg

### Фаза 4 — Branch 3: Текст
- [ ] OpenSCAD + LM Studio
- [ ] TRELLIS (опционально)

### Фаза 5 — Редактирование
- [ ] LLM planner (LM Studio API)
- [ ] Text/photo edit executors

### Фаза 6 — Полировка
- [ ] Task Scheduler автозапуск
- [ ] Документация

---

## Фаза 0.5 — Скрипты повторного развёртывания ✅

**Цель:** воспроизводимая установка «с нуля» за 2 команды.

### Структура `deploy/`

```
deploy/
├── README.md                    # Подробная инструкция
├── deploy.config.json           # host, user, ports (не в git)
├── deploy.config.example.json   # шаблон
├── lib/
│   └── Remote.ps1               # SSH helpers (encoded PowerShell)
└── scripts/
    ├── 01-bootstrap-ssh-server.ps1
    ├── 02-copy-ssh-key.ps1
    ├── 03-install-server.ps1
    ├── 04-deploy-remote.ps1
    ├── 05-verify-deployment.ps1
    └── 06-redeploy.ps1
deploy.ps1                       # точка входа с Z13
```

### Сценарии

| Сценарий | Команда |
|----------|---------|
| Первая установка | bootstrap (server) → `deploy.ps1 deploy` |
| Обновить зависимости | `deploy.ps1 redeploy` |
| Проверить состояние | `deploy.ps1 verify` |
| Новый сервер | изменить `deploy.config.json` → bootstrap → deploy |

### Планируемые скрипты (следующая итерация)

| Скрипт | Назначение |
|--------|------------|
| `07-deploy-app.ps1` | Только код приложения (без PyTorch) |
| `08-start-services.ps1` | Запуск MeshForge API + проверка LM Studio |
| `09-backup-projects.ps1` | Бэкап `C:\AI\mesh-forge\projects` |
| `10-restore-projects.ps1` | Восстановление проектов |

### Критерии готовности redeploy

- [x] `deploy.config.json` с параметрами машин
- [x] Bootstrap SSH + firewall документирован
- [x] Установка по SSH без ручного копирования файлов
- [x] `06-redeploy.ps1` для повторного развёртывания
- [x] `05-verify-deployment.ps1` проверяет SSH, CUDA, LM Studio
- [ ] `07-deploy-app.ps1` для обновления только UI
- [ ] CI-подобный чеклист в README

---

## Железо

| | DESKTOP-HOME | Alexey (Z13) |
|--|--------------|--------------|
| GPU | RTX 3070 Ti 8GB | Radeon 8060S |
| RAM | 32 GB | 32 GB |
| Роль | Compute + LM Studio | UI client |

## Документация

- [deploy/README.md](deploy/README.md) — развёртывание и redeploy
- [PLAN.md](PLAN.md) — этот файл
