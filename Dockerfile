FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System libs Docling / OpenCV / pdfplumber need
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    curl \
    sqlite3 \
    libgl1 \
    libglib2.0-0 \
    poppler-utils \
    libffi-dev \
 && rm -rf /var/lib/apt/lists/*

# CPU-only torch first: the LLM runs in Ollama, torch is only used for the
# CrossEncoder reranker + Docling layout models. Saves ~3GB of CUDA wheels.
RUN pip install --no-cache-dir torch torchvision \
      --index-url https://download.pytorch.org/whl/cpu

# Full stack (Docling included) -- NOT requirements.txt, which omits Docling
COPY requirements_sw_final.txt .
RUN pip install --no-cache-dir -r requirements_sw_final.txt

COPY . .
RUN chown -R 1000:1000 /app

EXPOSE 8000 8501
