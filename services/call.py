"""Continuous voice conversation with server-side call recording.

Differs from app.py in three ways that matter:

  * conversation state - each turn is appended to a history the LLM sees, so the
    agent remembers what was already said instead of answering in isolation
  * the browser decides when a turn ended (VAD on the mic), so the caller never
    presses anything after starting the call
  * every segment, caller and agent alike, is kept in order on the server and
    concatenated into one WAV on request - a real call recording rather than a
    pile of clips

Audio is normalised to 16 kHz mono throughout so segments concatenate without
resampling surprises at download time.
"""
import asyncio
import base64
import io
import json
import os
import secrets
import subprocess
import time
import uuid

import httpx
import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

VLLM = os.environ.get("VLLM_URL", "http://localhost:8000")
TTS_URL = os.environ.get("TTS_URL", "http://localhost:8002")
PASSWORD = os.environ.get("APP_PASSWORD", "")
STATIC = os.path.join(os.path.dirname(__file__), "static")
SR = 16000
MAX_TURNS = 40

SYSTEM = (
    "Aap VoAIs ke call center agent hain.\n\n"
    "SABSE ZAROORI NIYAM: Jawab HAMESHA Devanagari lipi (script) mein likhiye.\n"
    "Sahi:  नमस्ते, मैं आपकी क्या मदद कर सकता हूँ?\n"
    "Galat: Namaste, main aapki kya madad kar sakta hoon?\n"
    "Roman/English akshar bilkul mat likhiye, chahe user Roman ya English mein "
    "likhe. Angrezi shabd bhi Devanagari mein likhiye (email -> ईमेल, "
    "account -> अकाउंट).\n\n"
    "Phone par baat kar rahe hain, isliye ek ya do chhote vakya. Koi markdown, "
    "koi list, koi emoji nahi."
)

app = FastAPI(title="VoAIs call")
_sessions: set[str] = set()
_calls: dict[str, dict] = {}


def _auth(token: str) -> None:
    if token not in _sessions:
        raise HTTPException(401, "not authenticated")


def _call(cid: str) -> dict:
    c = _calls.get(cid)
    if c is None:
        raise HTTPException(404, "no such call")
    return c


def _decode_to_16k(raw: bytes) -> np.ndarray:
    """Whatever the browser sent -> float32 mono @16k."""
    p = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-i", "pipe:0", "-ar", str(SR),
         "-ac", "1", "-f", "wav", "pipe:1"],
        input=raw, capture_output=True,
    )
    if p.returncode != 0:
        raise HTTPException(400, f"audio decode failed: {p.stderr.decode()[:200]}")
    y, _ = sf.read(io.BytesIO(p.stdout), dtype="float32")
    return y


def _wav_bytes(y: np.ndarray, sr: int = SR) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


@app.post("/api/login")
async def login(password: str = Form(...)):
    if not PASSWORD:
        raise HTTPException(500, "APP_PASSWORD not set")
    if not secrets.compare_digest(password, PASSWORD):
        await asyncio.sleep(1.0)
        raise HTTPException(401, "wrong password")
    t = secrets.token_urlsafe(32)
    _sessions.add(t)
    return {"token": t}


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


@app.post("/api/call/start")
async def call_start(token: str = Form(...)):
    _auth(token)
    cid = uuid.uuid4().hex[:12]
    _calls[cid] = {"history": [], "segments": [], "started": time.time()}
    return {"call_id": cid}


@app.post("/api/call/turn")
async def call_turn(token: str = Form(...), call_id: str = Form(...),
                    file: UploadFile = File(...), tts: str = Form("1"),
                    nfe: int = Form(16)):
    _auth(token)
    call = _call(call_id)
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "empty audio")

    async def events():
        t0 = time.perf_counter()
        try:
            user_audio = _decode_to_16k(raw)
            if len(user_audio) < SR * 0.25:
                yield _sse({"stage": "skip", "reason": "too short"})
                return

            b64 = base64.b64encode(_wav_bytes(user_audio)).decode()
            t = time.perf_counter()
            body = {"model": "voais-llm", "max_tokens": 96, "temperature": 0.0,
                    "messages": [{"role": "user", "content": [
                        {"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64," + b64}},
                        {"type": "text", "text": "Transcribe this audio verbatim. Output only the transcript."}]}]}
            async with httpx.AsyncClient(timeout=90) as c:
                r = await c.post(f"{VLLM}/v1/chat/completions", json=body)
                r.raise_for_status()
                data = r.json()
                heard = data["choices"][0]["message"]["content"].strip()
                # a completion that ran to the cap is a degenerate loop, not speech
                if data["usage"]["completion_tokens"] >= 96:
                    yield _sse({"stage": "skip", "reason": "asr loop"})
                    return
            stt_ms = (time.perf_counter() - t) * 1000
            if not heard:
                yield _sse({"stage": "skip", "reason": "empty transcript"})
                return
            yield _sse({"stage": "stt", "ms": round(stt_ms), "text": heard})

            call["segments"].append(("user", user_audio))
            call["history"].append({"role": "user", "content": heard})
            del call["history"][:-MAX_TURNS]

            msgs = [{"role": "system", "content": SYSTEM}] + call["history"]
            t_llm = time.perf_counter()
            ttft, parts = None, []
            async with httpx.AsyncClient(timeout=90) as c:
                async with c.stream("POST", f"{VLLM}/v1/chat/completions",
                                    json={"model": "voais-llm", "stream": True,
                                          "max_tokens": 80, "temperature": 0.4,
                                          "messages": msgs}) as r:
                    async for line in r.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        p = line[6:]
                        if p == "[DONE]":
                            break
                        d = json.loads(p)["choices"][0]["delta"].get("content")
                        if not d:
                            continue
                        if ttft is None:
                            ttft = (time.perf_counter() - t_llm) * 1000
                            yield _sse({"stage": "ttft", "ms": round(ttft),
                                        "wait_ms": round(stt_ms + ttft)})
                        parts.append(d)
                        yield _sse({"stage": "token", "text": d})

            reply = "".join(parts).strip()
            call["history"].append({"role": "assistant", "content": reply})
            yield _sse({"stage": "llm_done", "ms": round((time.perf_counter() - t_llm) * 1000),
                        "text": reply})

            if tts == "1" and reply:
                # Synthesise sentence by sentence. IndicF5 is non-autoregressive,
                # so it cannot stream within an utterance - but it can start
                # speaking sentence one while sentence two is still being made,
                # which is where the perceived latency actually drops.
                first_ms = None
                t_tts = time.perf_counter()
                try:
                    async with httpx.AsyncClient(timeout=120) as c:
                        for idx, chunk in enumerate(_sentences(reply)):
                            t_c = time.perf_counter()
                            r = await c.post(f"{TTS_URL}/synthesize",
                                             json={"text": chunk, "language": "hi",
                                                   "nfe": nfe})
                            r.raise_for_status()
                            agent = _decode_to_16k(r.content)
                            call["segments"].append(("agent", agent))
                            if first_ms is None:
                                first_ms = round((time.perf_counter() - t_tts) * 1000)
                                yield _sse({"stage": "tts_first", "ms": first_ms})
                            yield _sse({"stage": "tts_chunk", "i": idx, "text": chunk,
                                        "ms": round((time.perf_counter() - t_c) * 1000),
                                        "audio": base64.b64encode(_wav_bytes(agent)).decode()})
                    yield _sse({"stage": "tts", "ms": round((time.perf_counter() - t_tts) * 1000),
                                "first_ms": first_ms})
                except Exception as exc:
                    yield _sse({"stage": "tts_error", "error": str(exc)[:180]})

            yield _sse({"stage": "total", "ms": round((time.perf_counter() - t0) * 1000)})
        except Exception as exc:
            yield _sse({"stage": "error", "error": str(exc)[:300]})

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/call/recording")
async def recording(token: str, call_id: str):
    """Both sides, in the order they were spoken, as one WAV."""
    _auth(token)
    call = _call(call_id)
    if not call["segments"]:
        raise HTTPException(404, "nothing recorded yet")
    gap = np.zeros(int(SR * 0.25), dtype="float32")
    joined = np.concatenate(
        [np.concatenate([a, gap]) for _, a in call["segments"]]).astype("float32")
    peak = float(np.abs(joined).max()) or 1.0
    if peak > 1.0:
        joined = joined / peak
    return Response(
        content=_wav_bytes(joined),
        media_type="audio/wav",
        headers={"Content-Disposition": f'attachment; filename="voais-call-{call_id}.wav"'},
    )


@app.get("/api/call/transcript")
async def transcript(token: str, call_id: str):
    _auth(token)
    return {"call_id": call_id, "turns": _call(call_id)["history"]}


def _sentences(text: str, max_chars: int = 140, first_max: int = 45) -> list[str]:
    """Split for speech, deliberately cutting the FIRST piece short.

    Synthesis cost scales with chunk length, and the caller only waits for
    chunk one - everything after it is generated while they are already
    listening. So the first chunk is broken at the earliest clause boundary
    (comma or sentence end) rather than the natural sentence, which is what
    pulls time-to-first-audio down. Later chunks stay long enough to keep
    prosody intact.
    """
    parts, buf = [], ""
    limit = first_max
    for ch in text:
        buf += ch
        stripped = buf.strip()
        ends = ch in "।?!."
        clause = ch in ",;" and not parts          # only the first piece may cut on a comma
        if (ends or clause) and len(stripped) >= 12 and len(stripped) >= limit * 0.4:
            parts.append(stripped)
            buf = ""
            limit = max_chars
        elif len(buf) >= limit and ch == " ":
            parts.append(stripped)
            buf = ""
            limit = max_chars
    if buf.strip():
        if parts and len(buf.strip()) < 12:
            parts[-1] += " " + buf.strip()
        else:
            parts.append(buf.strip())
    return parts or [text]


def _sse(o: dict) -> str:
    return f"data: {json.dumps(o, ensure_ascii=False)}\n\n"


@app.get("/")
async def index():
    return FileResponse(os.path.join(STATIC, "call.html"))


app.mount("/static", StaticFiles(directory=STATIC), name="static")
