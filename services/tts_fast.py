"""IndicF5 TTS on :8002, with the per-call overhead taken out.

`INF5Model.forward` redoes fixed work on every request: it silence-trims the
reference clip through pydub, writes a temp file, reloads that file with
torchaudio, and calls `.to(device)` on both the model and the vocoder. None of
that depends on the text being spoken. Measured on an L4 it dominated a short
utterance - a four-word phrase cost about as much as a full sentence.

So `forward` is bypassed. Everything invariant is done once at startup and
`infer_batch_process` is called directly, which also makes `nfe_step` reachable:
it is a def-time default inside `infer_process`, so setting the module global
after import silently does nothing.

The weight load is verified rather than trusted - see README for why.
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
NFE = int(os.environ.get("F5_NFE_STEP", "8"))       # 32 is upstream default
SR = 24000
REF_WAV = os.path.join(REPO, "prompts", "PAN_F_HAPPY_00001.wav")
REF_TEXT = ("ਭਹੰਪੀ ਵਿੱਚ ਸਮਾਰਕਾਂ ਦੇ ਭਵਨ ਨਿਰਮਾਣ ਕਲਾ ਦੇ ਵੇਰਵੇ ਗੁੰਝਲਦਾਰ ਅਤੇ ਹੈਰਾਨ ਕਰਨ ਵਾਲੇ ਹਨ, "
            "ਜੋ ਮੈਨੂੰ ਖੁਸ਼ ਕਰਦੇ  ਹਨ।")

app = FastAPI(title="VoAIs TTS (IndicF5, cached ref)")
_lock = threading.Lock()
_state = {}


def _boot():
    import torch
    import torchaudio
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    sys.path.insert(0, REPO)
    spec = importlib.util.spec_from_file_location("indicf5_local", os.path.join(REPO, "model.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    from f5_tts.infer.utils_infer import (chunk_text, infer_batch_process,
                                          preprocess_ref_audio_text)
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(REPO_ID, trust_remote_code=True)
    # Construct directly rather than via from_pretrained. transformers runs
    # __init__ inside a meta-device context for lazy loading, and this __init__
    # builds a real Vocos vocoder and moves it to CUDA - which cannot be done
    # from meta. Calling the constructor sidesteps that entirely, and since
    # __init__ already loads both the vocoder and the CFM checkpoint itself,
    # from_pretrained was never adding anything except the clobbering re-init.
    model = mod.INF5Model(cfg)

    # __init__ loads the CFM through the patched load_checkpoint, but verify it
    # rather than trust it: strict=False failures here are silent and produce
    # confident noise instead of speech.
    ckpt = hf_hub_download(REPO_ID, filename="model.safetensors")
    device = next(model.ema_model.parameters()).device
    tensors = {}
    with safe_open(ckpt, framework="pt") as f:
        for k in f.keys():
            if k.startswith("ema_model._orig_mod."):
                tensors[k[len("ema_model._orig_mod."):]] = f.get_tensor(k).to(device)
    model.ema_model.load_state_dict(tensors, strict=False)
    live = dict(model.ema_model.named_parameters())
    good = sum(1 for n, w in tensors.items()
               if n in live and torch.allclose(w.float().cpu(), live[n].detach().float().cpu(), atol=1e-5))
    if good < 200:
        raise RuntimeError(f"weight verification failed: only {good} tensors match")

    dev = model.device
    model.ema_model.to(dev).eval()
    model.vocoder.to(dev).eval()

    ref_path, ref_text = preprocess_ref_audio_text(REF_WAV, REF_TEXT)
    ref_audio, ref_sr = torchaudio.load(ref_path)
    ref_secs = ref_audio.shape[-1] / ref_sr
    max_chars = int(len(ref_text.encode("utf-8")) / ref_secs * (25 - ref_secs))

    _state.update(model=model, ref=(ref_audio, ref_sr), ref_text=ref_text,
                  max_chars=max_chars, device=dev, verified=good,
                  chunk_text=chunk_text, infer=infer_batch_process)
    return good


def _synth(text: str, nfe: int) -> np.ndarray:
    s = _state
    batches = s["chunk_text"](text, max_chars=s["max_chars"])
    with _lock:
        wav, _, _ = s["infer"](
            s["ref"], s["ref_text"], batches,
            s["model"].ema_model, s["model"].vocoder,
            mel_spec_type="vocos", nfe_step=nfe, device=s["device"],
            progress=None,
        )
    a = np.asarray(wav, dtype=np.float32)
    peak = float(np.abs(a).max()) or 1.0
    if peak > 1.0:
        a = a / peak
    return a


class Req(BaseModel):
    text: str
    language: str = "hi"
    nfe: int | None = None


@app.on_event("startup")
def startup():
    try:
        t = time.time()
        good = _boot()
        print(f"indicf5 ready in {time.time()-t:.1f}s, {good} tensors verified, nfe={NFE}", flush=True)
        t = time.time()
        _synth("नमस्ते जी", NFE)
        print(f"warmup {time.time()-t:.1f}s", flush=True)
    except Exception as exc:
        import traceback
        print("BOOT FAILED:", exc, flush=True)
        traceback.print_exc()


@app.get("/health")
def health():
    return {"status": "ok" if _state.get("model") is not None else "loading",
            "nfe": NFE, "sample_rate": SR, "verified": _state.get("verified")}


@app.post("/synthesize")
def synthesize(req: Req):
    if not req.text.strip():
        raise HTTPException(400, "empty text")
    if not _state.get("model"):
        raise HTTPException(503, "loading")
    try:
        audio = _synth(req.text, req.nfe or NFE)
    except Exception as exc:
        raise HTTPException(500, f"synthesis failed: {exc}") from exc
    buf = io.BytesIO()
    sf.write(buf, audio, SR, format="WAV", subtype="PCM_16")
    return Response(content=buf.getvalue(), media_type="audio/wav")
