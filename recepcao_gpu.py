"""
Recepção HTTP que roda DENTRO da máquina GPU do RunPod.

O Painel do Operador (na máquina do Emerson) conversa só com estes endereços:

  GET  /saude                  -> {"pronto": true, "modelos": "carregados"}
  POST /trabalho               -> {"id": "..."}      (não espera: devolve na hora)
  GET  /trabalho/{id}          -> estado + progresso + log
  GET  /trabalho/{id}/resultado-> JSON completo (segmentos, palavras, falantes)
  POST /encerrar               -> destrói a própria máquina

Toda chamada exige o cabeçalho X-Segredo igual à variável de ambiente SEGREDO.
Se ficar MINUTOS_INATIVIDADE sem trabalho, a máquina se autodestrói.
"""
import os
import time
import uuid
import glob
import shutil
import subprocess
import threading
import tempfile
import requests
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

SEGREDO = os.environ.get("SEGREDO", "")
PORTA = int(os.environ.get("PORTA", "8000"))
IDIOMA = os.environ.get("IDIOMA", "pt")
MODELO_WHISPER = os.environ.get("MODELO_WHISPER", "large-v3")
MINUTOS_INATIVIDADE = int(os.environ.get("MINUTOS_INATIVIDADE", "20"))
HF_TOKEN = os.environ.get("HF_TOKEN", "")
RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY", "")
POD_ID = os.environ.get("RUNPOD_POD_ID", "")

app = FastAPI()
trabalhos = {}
trava = threading.Lock()
ultima_atividade = time.time()
modelos = {"whisper": None, "diarizador": None, "pronto": False}


# ---------------------------------------------------------------- utilidades
def conferir(segredo):
    if not SEGREDO or segredo != SEGREDO:
        raise HTTPException(status_code=401, detail="segredo invalido")


def marcar_atividade():
    global ultima_atividade
    ultima_atividade = time.time()


def destruir_maquina(motivo="fim"):
    print(f"[recepcao] destruindo maquina ({motivo})", flush=True)
    if not (RUNPOD_API_KEY and POD_ID):
        print("[recepcao] sem RUNPOD_API_KEY/POD_ID: nao consigo me destruir", flush=True)
        return False
    try:
        r = requests.delete(
            f"https://rest.runpod.io/v1/pods/{POD_ID}",
            headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"},
            timeout=30,
        )
        print(f"[recepcao] terminate HTTP {r.status_code}", flush=True)
        return r.status_code < 300
    except Exception as e:
        print(f"[recepcao] falha ao destruir: {e}", flush=True)
        return False


def vigia_inatividade():
    while True:
        time.sleep(60)
        ocupado = any(t["estado"] in ("na_fila", "processando") for t in trabalhos.values())
        parado_min = (time.time() - ultima_atividade) / 60
        if not ocupado and parado_min >= MINUTOS_INATIVIDADE:
            destruir_maquina(f"inatividade de {int(parado_min)} min")
            return


# ---------------------------------------------------------------- modelos
def carregar_modelos():
    import whisperx
    import torch

    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    tipo = "float16" if dispositivo == "cuda" else "int8"
    print(f"[recepcao] carregando whisper {MODELO_WHISPER} em {dispositivo}", flush=True)
    modelos["whisper"] = whisperx.load_model(
        MODELO_WHISPER, dispositivo, compute_type=tipo, language=IDIOMA
    )
    try:
        from whisperx.diarize import DiarizationPipeline
    except ImportError:
        from whisperx import DiarizationPipeline
    modelos["diarizador"] = DiarizationPipeline(
        use_auth_token=HF_TOKEN or None, device=dispositivo
    )
    modelos["dispositivo"] = dispositivo
    modelos["pronto"] = True
    print("[recepcao] modelos carregados", flush=True)


# ---------------------------------------------------------------- processamento
def eh_youtube(url):
    u = (url or "").lower()
    return "youtube.com" in u or "youtu.be" in u


def _yt_dlp(url, saida_modelo, cookies=None, log=None):
    """Roda o yt-dlp; devolve (ok, mensagem)."""
    cmd = ["python", "-m", "yt_dlp", "-x", "--audio-format", "mp3",
           "--no-playlist", "--no-progress", "-o", saida_modelo]
    if cookies:
        cmd += ["--cookies", cookies]
    cmd.append(url)
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    saida = (p.stdout or "") + (p.stderr or "")
    if log and saida.strip():
        for linha in saida.strip().splitlines()[-4:]:
            log("    yt-dlp: " + linha[:200])
    return p.returncode == 0, saida[-500:]


def baixar_audio(url, destino, cookies=None, log=None):
    """
    Podcast: baixa o arquivo direto.
    YouTube: usa o yt-dlp com o cookie que o painel mandou.

    O YouTube muda com frequência e quebra o yt-dlp. Por isso, se falhar,
    tentamos ATUALIZAR o yt-dlp e repetir uma vez — assim a imagem não
    precisa ser reconstruída toda vez que o YouTube mexe em alguma coisa.
    """
    if not eh_youtube(url):
        with requests.get(url, stream=True, timeout=300) as r:
            r.raise_for_status()
            with open(destino, "wb") as f:
                for pedaco in r.iter_content(1024 * 512):
                    f.write(pedaco)
        return destino

    pasta = tempfile.mkdtemp(prefix="yt_")
    modelo = os.path.join(pasta, "audio.%(ext)s")
    ok, msg = _yt_dlp(url, modelo, cookies, log)
    if not ok:
        if log:
            log("    yt-dlp falhou; atualizando e tentando de novo...")
        subprocess.run(["pip", "install", "-q", "--upgrade", "yt-dlp"],
                       capture_output=True, text=True, timeout=600)
        ok, msg = _yt_dlp(url, modelo, cookies, log)
    achados = glob.glob(os.path.join(pasta, "audio.*"))
    if not ok or not achados:
        shutil.rmtree(pasta, ignore_errors=True)
        raise RuntimeError("não consegui baixar o áudio do YouTube. "
                           "Provável cookie vencido ou bloqueio de IP. " + msg[-200:])
    shutil.move(achados[0], destino)
    shutil.rmtree(pasta, ignore_errors=True)
    return destino


def processar(tid):
    import whisperx

    t = trabalhos[tid]
    caminho = None
    cookies_arq = None
    try:
        t["estado"] = "processando"
        t["etapa"] = "baixando audio"
        t["log"].append("baixando audio")
        caminho = os.path.join(tempfile.gettempdir(), f"{tid}.audio")
        # O cookie do YouTube chega no pedido e é gravado só aqui dentro,
        # nesta máquina que será destruída em seguida. Nunca vai para a
        # imagem (que é pública) nem fica registrado no RunPod.
        if t.get("cookies"):
            cookies_arq = os.path.join(tempfile.gettempdir(), f"{tid}.cookies.txt")
            with open(cookies_arq, "w", encoding="utf-8") as f:
                f.write(t["cookies"])
        baixar_audio(t["url_audio"], caminho, cookies_arq,
                     log=lambda m: t["log"].append(m))

        t["etapa"] = "transcrevendo"
        t["log"].append("transcrevendo")
        audio = whisperx.load_audio(caminho)
        resultado = modelos["whisper"].transcribe(audio, batch_size=16)

        t["etapa"] = "alinhando palavras"
        t["log"].append("alinhando palavras")
        modelo_align, meta = whisperx.load_align_model(
            language_code=resultado.get("language", IDIOMA), device=modelos["dispositivo"]
        )
        resultado = whisperx.align(
            resultado["segments"], modelo_align, meta, audio,
            modelos["dispositivo"], return_char_alignments=False,
        )

        t["etapa"] = "diarizando"
        t["log"].append(f"diarizando ({t['participantes'] or 'auto'} vozes)")
        kwargs = {}
        if t["participantes"]:
            kwargs = {"min_speakers": t["participantes"], "max_speakers": t["participantes"]}
        diar = modelos["diarizador"](caminho, **kwargs)
        resultado = whisperx.assign_word_speakers(diar, resultado)

        vozes = sorted({s.get("speaker", "SPEAKER_?") for s in resultado["segments"]})
        t["resultado"] = {
            "idioma": IDIOMA,
            "modelo": MODELO_WHISPER,
            "motor": "whisperx_runpod",
            "vozes": vozes,
            "segments": resultado["segments"],
        }
        t["estado"] = "concluido"
        t["etapa"] = "pronto"
        t["log"].append(f"pronto: {len(resultado['segments'])} falas, {len(vozes)} vozes")
    except Exception as e:
        t["estado"] = "erro"
        t["erro"] = str(e)
        t["log"].append(f"ERRO: {e}")
        print(f"[recepcao] erro no trabalho {tid}: {e}", flush=True)
    finally:
        for lixo in (caminho, cookies_arq):
            if lixo and os.path.exists(lixo):
                os.remove(lixo)          # o cookie não sobrevive ao trabalho
        marcar_atividade()


# ---------------------------------------------------------------- rotas
class Pedido(BaseModel):
    url_audio: str
    participantes: int | None = None
    nome: str | None = None
    cookies: str | None = None      # conteúdo do cookies.txt (só YouTube)


@app.get("/saude")
def saude(x_segredo: str = Header(default="")):
    conferir(x_segredo)
    marcar_atividade()
    return {"pronto": modelos["pronto"], "pod": POD_ID}


@app.post("/trabalho")
def novo_trabalho(p: Pedido, x_segredo: str = Header(default="")):
    conferir(x_segredo)
    marcar_atividade()
    if not modelos["pronto"]:
        raise HTTPException(status_code=503, detail="modelos ainda carregando")
    tid = uuid.uuid4().hex[:12]
    trabalhos[tid] = {
        "id": tid, "estado": "na_fila", "etapa": "aguardando", "log": [],
        "url_audio": p.url_audio, "participantes": p.participantes,
        "nome": p.nome, "cookies": p.cookies,
        "resultado": None, "erro": None, "criado": time.time(),
    }

    def fila():
        with trava:  # uma transcrição por vez
            processar(tid)

    threading.Thread(target=fila, daemon=True).start()
    return {"id": tid}


@app.get("/trabalho/{tid}")
def status(tid: str, x_segredo: str = Header(default="")):
    conferir(x_segredo)
    marcar_atividade()
    t = trabalhos.get(tid)
    if not t:
        raise HTTPException(status_code=404, detail="trabalho nao encontrado")
    return {k: t[k] for k in ("id", "estado", "etapa", "log", "erro", "nome")}


@app.get("/trabalho/{tid}/resultado")
def resultado(tid: str, x_segredo: str = Header(default="")):
    conferir(x_segredo)
    marcar_atividade()
    t = trabalhos.get(tid)
    if not t:
        raise HTTPException(status_code=404, detail="trabalho nao encontrado")
    if t["estado"] != "concluido":
        raise HTTPException(status_code=409, detail=f"estado atual: {t['estado']}")
    return JSONResponse(t["resultado"])


@app.post("/encerrar")
def encerrar(x_segredo: str = Header(default="")):
    conferir(x_segredo)
    threading.Thread(target=destruir_maquina, args=("pedido do painel",), daemon=True).start()
    return {"destruindo": True}


if __name__ == "__main__":
    threading.Thread(target=carregar_modelos, daemon=True).start()
    threading.Thread(target=vigia_inatividade, daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=PORTA, log_level="warning")
