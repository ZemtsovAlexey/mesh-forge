# Photo → 3D Docker

Два образа: **TripoSR** и **Hunyuan3D-2mini** (shape-only, лучше качество на 8GB).

## Требования

- Docker Desktop (Windows) с WSL2 backend
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) в WSL2
- `docker run --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi` должен работать

## Сборка

### TripoSR

```powershell
cd C:\AI\mesh-forge
.\docker\triposr\build.ps1
```

### Hunyuan3D-2mini (рекомендуется)

```powershell
.\docker\hunyuan3d\build.ps1
```

Первая сборка тянет PyTorch runtime (~3 GB). Первый запуск Hunyuan скачает веса `tencent/Hunyuan3D-2mini` в `hf_cache` (~1–2 GB).

## Конфиг (`config.yaml`)

```yaml
docker:
  enabled: true
  triposr_image: meshforge/triposr:latest
  hunyuan_image: meshforge/hunyuan3d:latest
  hunyuan_model: tencent/Hunyuan3D-2mini
  hunyuan_subfolder: hunyuan3d-dit-v2-mini-turbo
  hunyuan_steps: 20
  hunyuan_octree: 256
  hunyuan_chunks: 8000
  hf_cache: C:/AI/mesh-forge/.cache/huggingface
photo:
  backend: hunyuan3d   # или triposr
```

В UI на вкладке «Фото» можно переключить модель на каждый запуск.

## Отключить Docker (только TripoSR fallback)

```yaml
docker:
  enabled: false
```

Тогда TripoSR использует локальный `paths.triposr` + `venv-triposr`. Hunyuan без Docker не поддерживается.

## Устранение неполадок

### `error getting credentials` / logon session

В `%USERPROFILE%\.docker\config.json` удалите строку `"credsStore": "desktop"` (бэкап: `config.json.bak`).
Скрипты `build.ps1` делают это автоматически.

**Важно:** не сохраняйте файл через `Set-Content -Encoding UTF8` — PowerShell добавляет BOM, и Docker Desktop не запустится (`invalid character 'ï'`). Используйте `build.ps1` или сохраняйте UTF-8 без BOM.

### Docker Desktop не стартует / WSL VHDX

1. Полностью выйти из Docker Desktop (трей → Quit)
2. `wsl --shutdown` в PowerShell
3. Запустить Docker Desktop снова
4. Если не помогло — перезагрузка Windows или Docker Desktop → Troubleshoot → Restart

### Повреждённый слой при pull (`corrupted -- incomplete deflate data`)

```powershell
docker builder prune -f
docker pull pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime
```

## Кэш моделей Hugging Face

Модели кэшируются в `hf_cache` на хосте — повторные запуски быстрее.

## Качество / VRAM

| Модель | VRAM (shape) | Заметки |
|--------|--------------|---------|
| Hunyuan3D-2mini turbo + flashvdm | ~6–8 GB | Лучше объём/детали, по умолчанию |
| TripoSR | ~6–8 GB | Быстрее, проще, но чаще «плоский» силуэт |

Hunyuan: только shape (без текстур) — texgen на 8GB не влезает.
TripoSR: marching cubes через **skimage** (`isosurface_skimage.py`), `--mc-resolution 320` на 8GB.
