"""IndicF5 TTS -> :8002

IndicF5 clones a reference voice: you supply one short reference WAV plus its
transcript, and every synthesis comes out in that voice. That is what gives a
calling agent a single consistent brand voice across all calls.

NOTE: the exact load/call signature is unverified — ai4bharat/IndicF5 is a gated
repo, so its usage docs could not be read before writing this. Expect to adjust
`_load()` and `_synth()` on first run against the real model card.
"""
import io
import os

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

MODEL_DIR = os.environ.get("INDICF5_DIR", "/mnt/models/IndicF5")
REF_WAV = os.environ.get("TTS_REF_WAV", "../data/samples/ref_voice.wav")
REF_TEXT = os.environ.get("TTS_REF_TEXT", "")
SAMPLE_RATE = 24000

app = FastAPI(title="VoAIs TTS (IndicF5)")
_model = None


def _load():
    global _model
    if _model is None:
        from transformers import AutoModel
        _model = AutoModel.from_pretrained(MODEL_DIR, trust_remote_code=True)
    return _model


def _synth(text: str) -> np.ndarray:
    model = _load()
    audio = model(text, ref_audio_path=REF_WAV, ref_text=REF_TEXT)
    audio = np.asarray(audio, dtype=np.float32)
    peak = np.abs(audio).max()
    if peak > 1.0:
        audio = audio / peak
    return audio


class SynthRequest(BaseModel):
    text: str
    language: str = "hi"


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_DIR, "sample_rate": SAMPLE_RATE}


@app.post("/synthesize")
def synthesize(req: SynthRequest):
    if not req.text.strip():
        raise HTTPException(400, "text is empty")
    try:
        audio = _synth(req.text)
    except Exception as exc:
        raise HTTPException(500, f"synthesis failed: {exc}") from exc
    buf = io.BytesIO()
    sf.write(buf, audio, SAMPLE_RATE, format="WAV")
    return Response(content=buf.getvalue(), media_type="audio/wav")
