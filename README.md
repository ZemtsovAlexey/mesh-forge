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

## Текущий статус

- ✅ SSH: `zemet@192.168.0.22`
- ✅ Серверный стек установлен (PyTorch CUDA, TripoSR, Blender, OpenSCAD)
- ⏳ LM Studio + модели (вручную)
- ⏳ MeshForge UI (в разработке)
