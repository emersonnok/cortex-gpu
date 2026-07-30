# Imagem do Cortex para transcrição+diarização em GPU (RunPod)
# Objetivo: modelos JÁ dentro da imagem -> boot quase instantâneo, zero custo fixo.
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/modelos/hf \
    MODELO_WHISPER=large-v3 \
    IDIOMA=pt

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.10 python3-pip ffmpeg git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python

# 1) Bibliotecas (camada pesada, fica em cache entre builds)
RUN pip install --no-cache-dir torch==2.1.2 torchaudio==2.1.2 \
      --index-url https://download.pytorch.org/whl/cu121
RUN pip install --no-cache-dir \
      whisperx==3.4.2 \
      fastapi==0.115.6 \
      "uvicorn[standard]==0.34.0" \
      requests==2.32.3 \
      huggingface_hub==0.27.1

# 2) Modelos embutidos (token do HF entra como SEGREDO de build: não fica na imagem)
COPY baixar_modelos.py /tmp/baixar_modelos.py
RUN --mount=type=secret,id=hf_token \
    HF_TOKEN="$(cat /run/secrets/hf_token)" python /tmp/baixar_modelos.py \
    && rm /tmp/baixar_modelos.py

# 3) Recepção HTTP
COPY recepcao_gpu.py /app/recepcao_gpu.py
WORKDIR /app
EXPOSE 8000

CMD ["python", "-u", "recepcao_gpu.py"]
