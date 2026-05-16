FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    python3.11-venv \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

# Instalacja zależności z uwzględnieniem kompatybilności XTTS-v2
RUN pip3 install --no-cache-dir torch==2.4.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
RUN pip3 install --no-cache-dir "transformers>=4.57,<5.0"
RUN pip3 install --no-cache-dir coqui-tts>=0.27.5,<0.29.0
RUN pip3 install --no-cache-dir .

EXPOSE 8000 7860