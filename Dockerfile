# FT-Bench vLLM Inference Server
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip git curl && \
    rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install vLLM
RUN pip install --no-cache-dir vllm==0.4.2

# Copy project
COPY . .
RUN pip install --no-cache-dir -e .

EXPOSE 8000

ENV MODEL_ID="najahaja/ftbench-qlora-llama3.2-3b-awq"
ENV QUANTIZATION="awq"
ENV MAX_MODEL_LEN="4096"
ENV GPU_MEMORY_UTILIZATION="0.90"

CMD ["python", "-m", "vllm.entrypoints.openai.api_server", \
     "--model", "${MODEL_ID}", \
     "--quantization", "${QUANTIZATION}", \
     "--max-model-len", "${MAX_MODEL_LEN}", \
     "--gpu-memory-utilization", "${GPU_MEMORY_UTILIZATION}", \
     "--host", "0.0.0.0", "--port", "8000"]
