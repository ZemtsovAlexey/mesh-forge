# TripoSR inference image (CUDA runtime). Build from repo root:
#   docker build -f docker/triposr/Dockerfile -t meshforge/triposr:latest .
#
# Marching cubes via skimage (patched isosurface) — no torchmcubes CUDA compile.

FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
WORKDIR /opt/TripoSR

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY docker/triposr/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

RUN git clone --depth 1 https://github.com/VAST-AI-Research/TripoSR.git /opt/TripoSR

# Replace torchmcubes helper with skimage (correct axes, no [2,1,0] swizzle)
COPY docker/triposr/isosurface_skimage.py /opt/TripoSR/tsr/models/isosurface.py

WORKDIR /opt/TripoSR
ENTRYPOINT ["python", "run.py"]
