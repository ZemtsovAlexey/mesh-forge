# TripoSR Docker

Изолированный образ для фото → 3D. Убирает необходимость в `venv-triposr` и shim'ах на хосте.

## Требования

- Docker Desktop (Windows) с WSL2 backend
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) в WSL2
- `docker run --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi` должен работать

## Сборка

```powershell
cd C:\AI\mesh-forge
.\docker\triposr\build.ps1
```

Или:

```powershell
docker build -f docker/triposr/Dockerfile -t meshforge/triposr:latest .
```

Первая сборка тянет PyTorch runtime (~3 GB). `torchmcubes` ставится как CPU-shim (без компиляции CUDA) — так Docker Desktop не падает по OOM.

## Конфиг (`config.yaml`)

```yaml
docker:
  enabled: true
  triposr_image: meshforge/triposr:latest
  hf_cache: C:/AI/mesh-forge/.cache/huggingface
```

При `docker.enabled: true` MeshForge вызывает:

```text
docker run --gpus all -v <work_dir>:/work -v <hf_cache>:/root/.cache/huggingface \
  meshforge/triposr:latest /work/input.png --output-dir /work/triposr ...
```

## Отключить Docker (fallback)

```yaml
docker:
  enabled: false
```

Тогда используется локальный `paths.triposr` + `venv-triposr` (legacy).

## Устранение неполадок

### `error getting credentials` / logon session

В `%USERPROFILE%\.docker\config.json` удалите строку `"credsStore": "desktop"` (бэкап: `config.json.bak`).
Скрипт `build.ps1` делает это автоматически.

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

Модель `stabilityai/TripoSR` кэшируется в `hf_cache` на хосте — повторные запуски быстрее.

## Качество меша

Marching cubes в образе — через **skimage** (файл `isosurface_skimage.py`), без CUDA-`torchmcubes`.
Нельзя оставлять свап осей `[2,1,0]` из оригинального TripoSR — он только для torchmcubes; со skimage меш получается сплющенным и «ступенчатым».
На 8GB по умолчанию: `--mc-resolution 320`.
