"""indic-parler-tts on :8002.

Chosen over IndicF5 because it is a plain transformers model with no vendored
training code, which is exactly what made IndicF5 unusable here (its bundled
f5_tts did not match the model.py shipped in the repo).

Voice identity comes from a text description rather than a reference clip. The
description is fixed at module level on purpose: a call centre agent must sound
like the same person on every call, and a varying description would not.
"""
import io
import os
import threading

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

MODEL_DIR = os.environ.get("PARLER_DIR", "/mnt/models/parler")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

VOICE = ("Divya speaks in a clear, warm and natural tone at a moderate pace. "
         "The recording is very close-sounding with no background noise.")

app = FastAPI(title="VoAIs TTS (indic-parler)")
_lock = threading.Lock()
_model = _tok = _desc_tok = None
_desc_ids = None


def _load():
    global _model, _tok, _desc_tok, _desc_ids
    if _model is not None:
        return
    from parler_tts import ParlerTTSForConditionalGeneration
    from transformers import AutoTokenizer
    _model = ParlerTTSForConditionalGeneration.from_pretrained(
        MODEL_DIR, torch_dtype=DTYPE).to(DEVICE).eval()
    _tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    _desc_tok = AutoTokenizer.from_pretrained(_model.config.text_encoder._name_or_path)
    # the speaker description never changes, so tokenise it once
    _desc_ids = _desc_tok(VOICE, return_tensors="pt").to(DEVICE)


class Req(BaseModel):
    text: str
    language: str = "hi"


@app.on_event("startup")
def startup():
    try:
        _load()
        print(f"parler ready on {DEVICE}, sr={_model.config.sampling_rate}", flush=True)
    except Exception as exc:
        print("parler load FAILED:", exc, flush=True)


@app.get("/health")
def health():
    return {"status": "ok" if _model is not None else "loading",
            "device": DEVICE,
            "sample_rate": getattr(getattr(_model, "config", None), "sampling_rate", None)}


@app.post("/synthesize")
def synthesize(req: Req):
    if not req.text.strip():
        raise HTTPException(400, "empty text")
    _load()
    prompt = _tok(req.text, return_tensors="pt").to(DEVICE)
    with _lock, torch.inference_mode():
        gen = _model.generate(
            input_ids=_desc_ids.input_ids,
            attention_mask=_desc_ids.attention_mask,
            prompt_input_ids=prompt.input_ids,
            prompt_attention_mask=prompt.attention_mask,
        )
    audio = gen.to(torch.float32).cpu().numpy().squeeze()
    peak = float(np.abs(audio).max()) or 1.0
    if peak > 1.0:
        audio = audio / peak
    buf = io.BytesIO()
    sf.write(buf, audio, _model.config.sampling_rate, format="WAV", subtype="PCM_16")
    return Response(content=buf.getvalue(), media_type="audio/wav")
