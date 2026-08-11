# MeshForge

3D-печать: фото / скан / текст → печатаемый STL. Локально, двухмашинная схема.

## Быстрый старт

```powershell
# На Z13 (клиент):
cd Projects\mesh-forge
copy deploy\deploy.config.example.json deploy\deploy.config.json
.\deploy.ps1 verify      # проверить сервер
.\deploy.ps1 redeploy    # переустановить стек
```

Подробнее: **[deploy/README.md](deploy/README.md)**

План разработки: **[PLAN.md](PLAN.md)**

## Запуск UI

**На сервере (DESKTOP-HOME):**
```powershell
copy config.yaml.example config.yaml
# заполните paths: blender, openscad, triposr
C:\AI\mesh-forge\venv\Scripts\python.exe server.py
```

Старый Gradio UI (deprecated): `python app.py`

**С Z13 — деплой кода на сервер:**
```powershell
.\deploy.ps1 deploy-app
```

UI: http://192.168.0.22:7860

## Текущий статус

- ✅ SSH: `zemet@192.168.0.22`
- ✅ Серверный стек (PyTorch CUDA, TripoSR, Blender, OpenSCAD)
- ✅ MeshForge API + веб-UI (FastAPI, Three.js 3D viewer)
- ⏳ LM Studio + модели (вручную на сервере)
