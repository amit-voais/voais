"""IndicF5 TTS on :8002.

The upstream repo does not run as published; three defects are patched on the
VM before this server can import it (see README). What is fixed here rather than
there: the model is warmed at startup, because the first synthesis pays lazy
init and a caller should never eat that.

Voice identity comes from a fixed reference clip, so every call is the same
speaker - which is the reason this was chosen over description-conditioned TTS.
"""
import importlib.util
import io
import os
import sys
import threading
import time

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

REPO = os.environ.get("INDICF5_DIR", "/mnt/models/IndicF5")
REPO_ID = "ai4bharat/IndicF5"
SR = 24000
REF_WAV = os.path.join(REPO, "prompts", "PAN_F_HAPPY_00001.wav")
REF_TEXT = ("ਭਹੰਪੀ ਵਿੱਚ ਸਮਾਰਕਾਂ ਦੇ ਭਵਨ ਨਿਰਮਾਣ ਕਲਾ ਦੇ ਵੇਰਵੇ ਗੁੰਝਲਦਾਰ ਅਤੇ ਹੈਰਾਨ ਕਰਨ ਵਾਲੇ ਹਨ, "
            "ਜੋ ਮੈਨੂੰ ਖੁਸ਼ ਕਰਦੇ  ਹਨ।")

app = FastAPI(title="VoAIs TTS (IndicF5)")
_lock = threading.Lock()
_model = None


def _load():
    """Build the model, then force the real weights in and prove they took.

    transformers' from_pretrained re-initialises the inner CFM after __init__
    has already loaded it, because the checkpoint keys carry a torch.compile
    _orig_mod prefix that no longer matches the uncompiled module. Loading
    inside __init__ therefore gets silently overwritten with random values and
    strict=False hides it - the model runs and emits confident noise. So the
    weights are injected here, after construction, and verified tensor by
    tensor. A mismatch raises rather than serving garbage audio.
    """
    global _model
    if _model is not None:
        return
    import torch
    from safetensors import safe_open
    from huggingface_hub import hf_hub_download

    sys.path.insert(0, REPO)
    spec = importlib.util.spec_from_file_location("indicf5_local", os.path.join(REPO, "model.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(REPO_ID, trust_remote_code=True)
    model = mod.INF5Model.from_pretrained(REPO_ID, config=cfg)

    ckpt = hf_hub_download(REPO_ID, filename="model.safetensors")
    device = next(model.ema_model.parameters()).device
    tensors = {}
    with safe_open(ckpt, framework="pt") as f:
        for key in f.keys():
            if not key.startswith("ema_model._orig_mod."):
                continue
            tensors[key[len("ema_model._orig_mod."):]] = f.get_tensor(key).to(device)
    missing, unexpected = model.ema_model.load_state_dict(tensors, strict=False)
    if unexpected:
        raise RuntimeError(f"checkpoint has {len(unexpected)} keys the model rejects")

    live = dict(model.ema_model.named_parameters())
    checked = bad = 0
    for name, want in tensors.items():
        have = live.get(name)
        if have is None:
            continue
        checked += 1
        if not torch.allclose(want.float().cpu(), have.detach().float().cpu(), atol=1e-5):
            bad += 1
    if checked < 200 or bad:
        raise RuntimeError(f"weight load failed: {checked} checked, {bad} mismatched")
    print(f"indicf5 weights verified: {checked} tensors match", flush=True)
    _model = model


def _synth(text: str) -> np.ndarray:
    with _lock:
        audio = _model(text, ref_audio_path=REF_WAV, ref_text=REF_TEXT)
    a = np.asarray(audio, dtype=np.float32)
    peak = float(np.abs(a).max()) or 1.0
    if peak > 1.0:
        a = a / peak
    return a


class Req(BaseModel):
    text: str
    language: str = "hi"


@app.on_event("startup")
def startup():
    try:
        t = time.time()
        _load()
        print(f"indicf5 loaded in {time.time()-t:.1f}s", flush=True)
        t = time.time()
        _synth("नमस्ते")          # warm the lazy paths off the caller's clock
        print(f"warmup done in {time.time()-t:.1f}s", flush=True)
    except Exception as exc:
        print("indicf5 load FAILED:", exc, flush=True)


@app.get("/health")
def health():
    return {"status": "ok" if _model is not None else "loading", "sample_rate": SR}


@app.post("/synthesize")
def synthesize(req: Req):
    if not req.text.strip():
        raise HTTPException(400, "empty text")
    if _model is None:
        raise HTTPException(503, "model still loading")
    try:
        audio = _synth(req.text)
    except Exception as exc:
        raise HTTPException(500, f"synthesis failed: {exc}") from exc
    buf = io.BytesIO()
    sf.write(buf, audio, SR, format="WAV", subtype="PCM_16")
    return Response(content=buf.getvalue(), media_type="audio/wav")
