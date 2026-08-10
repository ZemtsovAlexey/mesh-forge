# MeshForge — развёртывание

Инструкция по первичной установке и **повторному развёртыванию** стека на compute-сервере.

## Архитектура

| Машина | Роль | IP |
|--------|------|-----|
| **Alexey** (ROG Flow Z13) | Клиент: UI, Blender, слайсер | локально |
| **DESKTOP-HOME** | Сервер: LM Studio, TripoSR, PyTorch | `192.168.0.22` |

SSH-пользователь на сервере: **`zemet`** (не путать с именем на Z13).

## Быстрый старт

### 1. Настройка конфига (один раз)

```powershell
cd C:\Users\ZemtsovAlexey\Projects\mesh-forge
copy deploy\deploy.config.example.json deploy\deploy.config.json
# Отредактируйте host / user при необходимости
```

### 2. Bootstrap SSH (один раз, на DESKTOP-HOME, Admin)

```powershell
powershell -ExecutionPolicy Bypass -File deploy\scripts\01-bootstrap-ssh-server.ps1
```

Или вручную добавить ключ в `administrators_authorized_keys` (см. раздел «Устранение неполадок»).

### 3. Проверка SSH с Z13

```powershell
ssh zemet@192.168.0.22 hostname
# Ожидается: DESKTOP-HOME
```

### 4. Установка серверного стека

```powershell
.\deploy.ps1 deploy
# или
.\deploy\scripts\04-deploy-remote.ps1
```

Установка идёт 15–30 минут. Лог на сервере: `C:\AI\install-meshforge.log`

### 5. LM Studio (вручную на DESKTOP-HOME)

1. Установить https://lmstudio.ai/
2. Скачать `Qwen2.5-VL-7B-Instruct` (Q4)
3. Local Server → Start → порт `1234`

### 6. Проверка

```powershell
.\deploy.ps1 verify
```

---

## Скрипты

| Скрипт | Где запускать | Назначение |
|--------|---------------|------------|
| `deploy.ps1` | Z13 | Точка входа (см. ниже) |
| `01-bootstrap-ssh-server.ps1` | DESKTOP-HOME (Admin) | SSH + firewall + ключ клиента |
| `02-copy-ssh-key.ps1` | Z13 | Копировать SSH-ключ (пароль один раз) |
| `03-install-server.ps1` | DESKTOP-HOME | Установка ПО (обычно через 04) |
| `04-deploy-remote.ps1` | Z13 | Загрузка + запуск установки по SSH |
| `05-verify-deployment.ps1` | Z13 | Проверка SSH, CUDA, LM Studio |
| `06-redeploy.ps1` | Z13 | **Повторное развёртывание** всего стека |

### deploy.ps1 — команды

```powershell
.\deploy.ps1 copy-key   # скопировать SSH-ключ
.\deploy.ps1 deploy     # установить/обновить сервер
.\deploy.ps1 verify     # проверить
.\deploy.ps1 redeploy   # полный redeploy
```

---

## Повторное развёртывание (redeploy)

Когда нужно переустановить сервер с нуля или после смены железа:

```powershell
# 1. Убедиться что SSH работает
ssh zemet@192.168.0.22 hostname

# 2. Полный redeploy
.\deploy.ps1 redeploy

# 3. Дождаться завершения (лог на сервере)
ssh zemet@192.168.0.22 "powershell -Command Get-Content C:\AI\install-meshforge.log -Tail 10"

# 4. Проверка
.\deploy.ps1 verify
```

Что делает redeploy:
- загружает актуальные `03-install-server.ps1` и `requirements-server.txt`
- запускает установку в фоне на сервере
- не трогает SSH-ключи и firewall (bootstrap не повторяется)

---

## Что устанавливается на сервер

| Компонент | Путь |
|-----------|------|
| Python venv | `C:\AI\mesh-forge\venv` |
| config.yaml | `C:\AI\mesh-forge\config.yaml` |
| TripoSR | `C:\AI\TripoSR` |
| Проекты | `C:\AI\mesh-forge\projects` |
| Лог установки | `C:\AI\install-meshforge.log` |

Пакеты: Git, Python 3.11, Blender, OpenSCAD, PyTorch (CUDA), trimesh, pymeshlab, open3d, rembg, gradio.

---

## Порты (firewall)

| Порт | Сервис |
|------|--------|
| 22 | SSH |
| 1234 | LM Studio API |
| 7860 | MeshForge UI (будущее) |

---

## Устранение неполадок

### SSH: Permission denied

1. Проверьте **имя пользователя** на сервере: `whoami` → скорее всего `zemet`
2. Для **администраторов** ключ должен быть в:
   `C:\ProgramData\ssh\administrators_authorized_keys`
3. Права на admin-keys:
   ```powershell
   icacls C:\ProgramData\ssh\administrators_authorized_keys /inheritance:r /grant "*S-1-5-32-544:(F)" /grant "SYSTEM:F"
   ```

### SCP зависает

Скрипт `04-deploy-remote.ps1` передаёт файлы через **base64 + SSH** (обход зависания scp).

### SSH выполняет bash вместо PowerShell

На сервере default shell может быть Git Bash. Скрипты используют `powershell.exe -EncodedCommand`.

### CUDA False после установки

Переустановите PyTorch:
```powershell
C:\AI\mesh-forge\venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## Следующие этапы (см. PLAN.md)

- [ ] Развернуть код MeshForge (`app.py`, UI)
- [ ] systemd/Task Scheduler для автозапуска
- [ ] Скрипт `07-deploy-app.ps1` — только обновление кода без переустановки PyTorch
