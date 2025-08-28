FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 espeak-ng ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Set HF cache directories with write permissions
ENV HF_HOME=/data/huggingface
ENV TRANSFORMERS_CACHE=/data/huggingface/hub
RUN mkdir -p $HF_HOME && chmod -R 777 $HF_HOME
RUN mkdir -p $TRANSFORMERS_CACHE && chmod -R 777 $TRANSFORMERS_CACHE

# Whisper cache path
ENV XDG_CACHE_HOME=/data/.cache
RUN mkdir -p /data/.cache/whisper && chmod -R 777 /data/.cache

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -m spacy download en_core_web_sm

COPY app ./app

# Copy your Whisper safetensors
COPY app/models/whisper-small /data/.cache/whisper
RUN chmod -R 777 /data/.cache/whisper

ENV PORT=7860
EXPOSE 7860

CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT}
