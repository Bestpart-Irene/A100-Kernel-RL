# KernelForge-OpenEnv Docker Image
# CUDA 12.1 with H100 support for Modal deployment
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    pkg-config \
    libssl-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -s /usr/bin/python3.12 /usr/bin/python

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN uv pip install --system -r /tmp/requirements.txt \
    && uv pip install --system "cupy-cuda12x>=14.0"

# Create application directory
WORKDIR /app

# Copy application code
COPY . /app/

# Create directories for outputs and cache
RUN mkdir -p /app/outputs /app/cache /app/datasets

# Set permissions
RUN chmod +x /app/demo/streamlit_demo.py
RUN chmod +x /app/training/*.py
RUN chmod +x /app/datasets/*.py
RUN chmod +x /app/verification/*.py

# Expose ports for Streamlit demo and OpenEnv server
EXPOSE 8501
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import modal, cupy, networkx; print('OK')" || exit 1

# Default command — env-switchable: KERNELFORGE_MODE=server for OpenEnv HTTP server
CMD ["sh", "-c", "if [ \"$KERNELFORGE_MODE\" = 'server' ]; then python -m uvicorn openenv_env.server.app:app --host 0.0.0.0 --port 8000; else python demo/streamlit_demo.py; fi"]
