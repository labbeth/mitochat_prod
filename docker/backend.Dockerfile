FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# ------------------------------------------------------------------------------------
# System dependencies
# ------------------------------------------------------------------------------------
RUN apt-get update && apt-get install -y \
    git wget curl nano \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------------------------------
# Python installation
# ------------------------------------------------------------------------------------
RUN apt-get update && apt-get install -y python3 python3-pip python3-venv

WORKDIR /app

# ------------------------------------------------------------------------------------
# Copy backend code
# ------------------------------------------------------------------------------------
COPY scripts/ ./scripts/
COPY prompts/ ./prompts/
COPY assets/ ./assets/
COPY requirements.backend.txt .

# ------------------------------------------------------------------------------------
# Install Python dependencies
# ------------------------------------------------------------------------------------
RUN pip3 install --upgrade pip

# 1) Install PyTorch with CUDA 12.1
RUN pip3 install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# 2) Install remaining deps
RUN pip3 install -r requirements.backend.txt

# ------------------------------------------------------------------------------------
# vLLM installation (GPU-accelerated)
# ------------------------------------------------------------------------------------
RUN pip3 install vllm

# ------------------------------------------------------------------------------------
# Expose backend port
# ------------------------------------------------------------------------------------
EXPOSE 9000

# ------------------------------------------------------------------------------------
# Launch FastAPI backend
# ------------------------------------------------------------------------------------
CMD ["uvicorn", "scripts.fastapi_backend:app", "--host", "0.0.0.0", "--port", "9000"]
