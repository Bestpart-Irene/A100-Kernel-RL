FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Python 3.12
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3.12-dev python3-pip && \
    ln -sf /usr/bin/python3.12 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.12 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

RUN python -m ensurepip --upgrade && \
    python -m pip install --no-cache-dir --upgrade pip

# Python dependencies (matches Modal cuda_image)
RUN pip install --no-cache-dir \
    cupy-cuda12x>=14.0 \
    networkx>=3.0 \
    numpy>=1.26 \
    scipy>=1.12 \
    torch>=2.4 \
    ninja>=1.11 \
    pydantic>=2.4.0 \
    fastapi>=0.115 \
    uvicorn[standard]>=0.34

WORKDIR /app

# Copy eval service and dependencies
COPY eval_service/ eval_service/
COPY verification/ verification/
COPY openenv_env/ openenv_env/
COPY kernels/ kernels/

ENV PYTHONPATH=/app
ENV KERNELFORGE_TARGET_GPU=A100
ENV KERNELFORGE_TARGET_ARCH=sm_80

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8080/health').raise_for_status()" || exit 1

CMD ["uvicorn", "eval_service.app:app", "--host", "0.0.0.0", "--port", "8080"]
