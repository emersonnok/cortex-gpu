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

# 1) Torch da CUDA 12.1 (mesma da imagem base).
#    A versão 2.5.1 já satisfaz o "torch>=2.5.1" do whisperx, então o pip
#    NÃO troca por uma versão CUDA 13 no passo seguinte.
RUN pip install --no-cache-dir torch==2.5.1 torchaudio==2.5.1 \
      --index-url https://download.pytorch.org/whl/cu121

# 2) WhisperX e o servidor
RUN pip install --no-cache-dir \
      whisperx==3.4.2 \
      fastapi==0.115.6 \
      "uvicorn[standard]==0.34.0" \
      requests==2.32.3 \
      huggingface_hub==0.27.1

# 3) Trava de segurança: se o torch não for o da CUDA 12.1, o build FALHA aqui
#    (melhor descobrir agora, de graça, do que na GPU paga).
RUN python -c "import torch, sys; \
print('torch instalado:', torch.__version__); \
sys.exit(0 if 'cu121' in torch.__version__ else 1)"

# 4) Modelos embutidos (token do HF entra como SEGREDO de build: não fica na imagem)
COPY baixar_modelos.py /tmp/baixar_modelos.py
RUN --mount=type=secret,id=hf_token \
    HF_TOKEN="$(cat /run/secrets/hf_token)" python /tmp/baixar_modelos.py \
    && rm /tmp/baixar_modelos.py

# 5) Recepção HTTP
COPY recepcao_gpu.py /app/recepcao_gpu.py
WORKDIR /app
EXPOSE 8000

CMD ["python", "-u", "recepcao_gpu.py"]
