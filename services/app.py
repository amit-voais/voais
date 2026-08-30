"""VoAIs voice-agent test harness.

Serves the test UI and the API from one origin so the browser never has to reach
across hosts. Reached over an IAP tunnel, so the VM keeps no public IP.

Every stage is timed and the timings are streamed to the browser as they happen,
which is the point of this app: to produce a real latency baseline instead of a
guess. Stage boundaries reported to the client:

    stt        audio in -> transcript
    ttft       transcript sent -> first LLM token          <- the number that matters
    llm        first token -> last token
    tts        full text -> playable audio
    total      audio received -> audio ready
"""
import asyncio
import base64
import io
import json
import os
import secrets
import subprocess
import time

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

VLLM = os.environ.get("VLLM_URL", "http://localhost:8000")
TTS_URL = os.environ.get("TTS_URL", "http://localhost:8002")
PASSWORD = os.environ.get("APP_PASSWORD", "")
STATIC = os.path.join(os.path.dirname(__file__), "static")

SYSTEM_PROMPT = (
    "Aap ek Indian call center agent hain. Hindi mein chhota, natural jawab dijiye. "
    "Ek ya do vakya se zyada nahi. Koi bullet points nahi, koi markdown nahi."
)

app = FastAPI(title="VoAIs test harness")
_sessions: set[str] = set()


def _check(token: str | None) -> None:
    if not token or token not in _sessions:
        raise HTTPException(401, "not authenticated")


def _to_wav16k(raw: bytes) -> bytes:
    """Browsers hand us webm/opus; the model wants 16 kHz mono WAV."""
    p = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-i", "pipe:0", "-ar", "16000",
         "-ac", "1", "-f", "wav", "pipe:1"],
        input=raw, capture_output=True,
    )
    if p.returncode != 0:
        raise HTTPException(400, f"audio decode failed: {p.stderr.decode()[:200]}")
    return p.stdout


@app.post("/api/login")
async def login(password: str = Form(...)):
    if not PASSWORD:
        raise HTTPException(500, "APP_PASSWORD not configured on the server")
    if not secrets.compare_digest(password, PASSWORD):
        await asyncio.sleep(1.0)          # blunt throttle on guessing
        raise HTTPException(401, "wrong password")
    token = secrets.token_urlsafe(32)
    _sessions.add(token)
    return {"token": token}


@app.get("/api/health")
async def health():
    out = {}
    async with httpx.AsyncClient(timeout=5) as c:
        for name, url in (("llm", f"{VLLM}/health"), ("tts", f"{TTS_URL}/health")):
            try:
                out[name] = (await c.get(url)).status_code == 200
            except Exception:
                out[name] = False
    return out


async def _transcribe(wav: bytes, t0: float) -> tuple[str, float]:
    b64 = base64.b64encode(wav).decode()
    body = {
        "model": "voais-llm", "max_tokens": 256, "temperature": 0.0,
        "messages": [{"role": "user", "content": [
            {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{b64}"}},
            {"type": "text", "text": "Transcribe this audio verbatim. Output only the transcript."},
        ]}],
    }
    async with httpx.AsyncClient(timeout=120) as c:
        r = await c.post(f"{VLLM}/v1/chat/completions", json=body)
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"].strip()
    return text, (time.perf_counter() - t0) * 1000


@app.post("/api/turn")
async def turn(token: str = Form(...), file: UploadFile = File(...), tts: str = Form("1")):
    """One full voice turn, streamed as SSE with a timing on every stage."""
    _check(token)
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "empty audio")

    async def events():
        t_start = time.perf_counter()
        try:
            wav = _to_wav16k(raw)
            yield _sse({"stage": "decode", "ms": round((time.perf_counter() - t_start) * 1000)})

            t = time.perf_counter()
            transcript, stt_ms = await _transcribe(wav, t)
            yield _sse({"stage": "stt", "ms": round(stt_ms), "text": transcript})

            body = {
                "model": "voais-llm", "stream": True, "max_tokens": 150, "temperature": 0.3,
                "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                             {"role": "user", "content": transcript}],
            }
            t_llm = time.perf_counter()
            ttft = None
            parts: list[str] = []
            async with httpx.AsyncClient(timeout=120) as c:
                async with c.stream("POST", f"{VLLM}/v1/chat/completions", json=body) as r:
                    async for line in r.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        payload = line[6:]
                        if payload == "[DONE]":
                            break
                        delta = json.loads(payload)["choices"][0]["delta"].get("content")
                        if not delta:
                            continue
                        if ttft is None:
                            ttft = (time.perf_counter() - t_llm) * 1000
                            yield _sse({"stage": "ttft", "ms": round(ttft)})
                        parts.append(delta)
                        yield _sse({"stage": "token", "text": delta})

            reply = "".join(parts).strip()
            yield _sse({"stage": "llm_done",
                        "ms": round((time.perf_counter() - t_llm) * 1000),
                        "text": reply})

            if tts == "1" and reply:
                t_tts = time.perf_counter()
                try:
                    async with httpx.AsyncClient(timeout=180) as c:
                        r = await c.post(f"{TTS_URL}/synthesize",
                                         json={"text": reply, "language": "hi"})
                        r.raise_for_status()
                        yield _sse({"stage": "tts",
                                    "ms": round((time.perf_counter() - t_tts) * 1000),
                                    "audio": base64.b64encode(r.content).decode()})
                except Exception as exc:
                    yield _sse({"stage": "tts_error", "error": str(exc)[:200]})

            yield _sse({"stage": "total", "ms": round((time.perf_counter() - t_start) * 1000)})
        except Exception as exc:
            yield _sse({"stage": "error", "error": str(exc)[:300]})

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


@app.get("/")
async def index():
    return FileResponse(os.path.join(STATIC, "index.html"))


app.mount("/static", StaticFiles(directory=STATIC), name="static")
