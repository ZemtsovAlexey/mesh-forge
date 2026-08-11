# Hunyuan3D-2mini shape-only (CUDA runtime). Build from repo root:
#   docker build -f docker/hunyuan3d/Dockerfile -t meshforge/hunyuan3d:latest .
#
# No texgen / custom_rasterizer — fits ~6–8 GB VRAM with turbo + flashvdm + mc.

FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HY3DGEN_DISABLE_TEX=1
ENV HF_HUB_DISABLE_XET=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY docker/hunyuan3d/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

RUN git clone --depth 1 https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git /opt/Hunyuan3D-2 \
    && pip install --no-cache-dir -e /opt/Hunyuan3D-2 --no-deps

# Avoid pulling pymeshlab via package __init__ (shape-only path)
COPY docker/hunyuan3d/shapegen_init.py /opt/Hunyuan3D-2/hy3dgen/shapegen/__init__.py
COPY docker/hunyuan3d/infer.py /opt/hunyuan/infer.py

WORKDIR /opt/Hunyuan3D-2
ENTRYPOINT ["python", "-u", "/opt/hunyuan/infer.py"]
