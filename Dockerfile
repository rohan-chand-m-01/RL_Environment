FROM python:3.11-slim

# Metadata for HuggingFace Spaces
LABEL maintainer="AI Workplace Simulator"
LABEL org.opencontainers.image.title="AI Workplace Simulator"
LABEL org.opencontainers.image.description="OpenEnv-compatible RL environment for real-world LLM evaluation"

WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Environment defaults (HF_TOKEN must be supplied at runtime)
ENV PYTHONPATH=/app
ENV API_BASE_URL=https://api-inference.huggingface.co/v1
ENV MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.3

# Expose Gradio port for HuggingFace Spaces
EXPOSE 7860

# HF Spaces entrypoint: launches FastAPI server
CMD ["python", "server.py"]
