"""OmniVoice TTS on :8002, drop-in for the IndicF5 server.

Same HTTP contract as tts_fast.py (/health, POST /synthesize with
{text, language, nfe}) so the experimental agent's TTS service needs no change. `nfe` maps
to OmniVoice's `num_step` - the diffusion step count, which is the only real
latency lever. Measured on an L4, one Hindi sentence:

    num_step   32 -> 1909 ms
               16 -> 1024 ms
                8 ->  574 ms
                4 ->  354 ms
                2 ->  243 ms

Text length barely matters (RTF 0.04 at step 4), so the cost is essentially
fixed per utterance. That is why the pipeline chunks by clause and why the
opening phrases are cached: the fixed term cannot be optimised away, only
paid in advance or amortised across a longer chunk.
"""
import io, os, threading, time
import numpy as np, soundfile as sf, torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

SR = 24000
DEFAULT_STEPS = int(os.environ.get("OMNI_STEPS", "4"))
REF_WAV = os.path.expanduser(os.environ.get("OMNI_REF", "~/refsil_0.2_6.1.wav"))
REF_TEXT = os.environ.get(
    "OMNI_REF_TEXT",
    "मेजर किन खाना भूमध्य सागर में सामान्य क्षेत्रों की तरह रोटी सब्जियों और मांस")

app = FastAPI(title="VoAIs TTS (OmniVoice)")
_lock = threading.Lock()
_state = {}


class Req(BaseModel):
    text: str
    language: str | None = "hi"
    nfe: int | None = None


@app.on_event("startup")
def boot():
    try:
        from omnivoice import OmniVoice
        from omnivoice.models.omnivoice import OmniVoiceGenerationConfig as GC
        m = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cuda:0",
                                      dtype=torch.float16)
        # first call pays lazy init; a caller should never eat that
        t0 = time.perf_counter()
        m.generate(text="नमस्ते।", ref_audio=REF_WAV, ref_text=REF_TEXT,
                   generation_config=GC(num_step=DEFAULT_STEPS))
        _state.update(model=m, GC=GC, warm_ms=round((time.perf_counter() - t0) * 1000))
        print(f"OmniVoice ready (warm {_state['warm_ms']} ms, steps={DEFAULT_STEPS})", flush=True)
    except Exception as exc:
        import traceback
        print("BOOT FAILED:", exc, flush=True)
        traceback.print_exc()


@app.get("/health")
def health():
    return {"status": "ok" if _state.get("model") else "loading",
            "steps": DEFAULT_STEPS, "sample_rate": SR,
            "warm_ms": _state.get("warm_ms")}


@app.post("/synthesize")
def synthesize(req: Req):
    if not req.text.strip():
        raise HTTPException(400, "empty text")
    if not _state.get("model"):
        raise HTTPException(503, "loading")
    steps = req.nfe or DEFAULT_STEPS
    try:
        with _lock:
            out = _state["model"].generate(
                text=req.text, ref_audio=REF_WAV, ref_text=REF_TEXT,
                generation_config=_state["GC"](num_step=steps))
    except Exception as exc:
        raise HTTPException(500, f"synthesis failed: {exc}") from exc
    a = out[0] if isinstance(out, (list, tuple)) else out
    a = np.asarray(a, dtype=np.float32)
    peak = float(np.abs(a).max()) or 1.0
    if peak > 1.0:
        a = a / peak
    buf = io.BytesIO()
    sf.write(buf, a, SR, format="WAV", subtype="PCM_16")
    return Response(content=buf.getvalue(), media_type="audio/wav")
