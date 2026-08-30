"""IndicConformer STT -> :8001

Fallback / comparison path only. The happy path feeds audio straight into
Gemma-4-12B. This server exists so tests/asr_ab_test.py can measure whether
Gemma actually matches a purpose-built Indic ASR — especially on 8 kHz
telephony audio, which neither model was trained on.

The repo ships ONNX (encoder.onnx, ctc_decoder.onnx, per-language
joint_post_net_<lang>.onnx) plus a TorchScript preprocessor — no .nemo
checkpoint, so the AI4Bharat NeMo fork is NOT needed.

NOTE: gated repo, usage docs unread. Adjust _transcribe() on first run against
the shipped model_onnx_1b_batched_rnnt.py.
"""
import os

import librosa
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

MODEL_DIR = os.environ.get("CONFORMER_DIR", "/mnt/models/indic-conformer-600m-multilingual")
TARGET_SR = 16000

app = FastAPI(title="VoAIs STT (IndicConformer)")
_sessions: dict[str, ort.InferenceSession] = {}


def _session(name: str) -> ort.InferenceSession:
    if name not in _sessions:
        path = os.path.join(MODEL_DIR, "assets", f"{name}.onnx")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        _sessions[name] = ort.InferenceSession(path, providers=providers)
    return _sessions[name]


def _load_audio(raw: bytes) -> np.ndarray:
    import io
    audio, sr = librosa.load(io.BytesIO(raw), sr=TARGET_SR, mono=True)
    return audio.astype(np.float32)


def _transcribe(audio: np.ndarray, language: str) -> str:
    enc = _session("encoder")
    feats = audio[None, :]
    lengths = np.array([feats.shape[1]], dtype=np.int64)
    enc_out = enc.run(None, {enc.get_inputs()[0].name: feats,
                             enc.get_inputs()[1].name: lengths})
    dec = _session(f"joint_post_net_{language}")
    out = dec.run(None, {dec.get_inputs()[0].name: enc_out[0]})
    return str(out[0])


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_DIR, "sample_rate": TARGET_SR}


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...), language: str = Form("hi")):
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "empty audio")
    try:
        audio = _load_audio(raw)
        text = _transcribe(audio, language)
    except FileNotFoundError as exc:
        raise HTTPException(400, f"no model for language '{language}': {exc}") from exc
    except Exception as exc:
        raise HTTPException(500, f"transcription failed: {exc}") from exc
    return {"text": text, "language": language, "duration_sec": round(len(audio) / TARGET_SR, 2)}
