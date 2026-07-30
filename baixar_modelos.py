"""Baixa os modelos DENTRO da imagem, durante o build. Roda sem GPU."""
import os
from huggingface_hub import snapshot_download

TOKEN = os.environ.get("HF_TOKEN") or None

REPOS = [
    "Systran/faster-whisper-large-v3",                  # transcrição
    "jonatasgrosman/wav2vec2-large-xlsr-53-portuguese",  # alinhamento de palavras (pt)
    "pyannote/speaker-diarization-3.1",                  # diarização
    "pyannote/segmentation-3.0",                         # dependência da diarização
    "pyannote/wespeaker-voxceleb-resnet34-LM",           # dependência da diarização
]

for repo in REPOS:
    print(f"--> baixando {repo}", flush=True)
    snapshot_download(repo_id=repo, token=TOKEN)

print("modelos embutidos com sucesso", flush=True)
