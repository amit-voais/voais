"""VoAIs streaming voice agent.

    mic ──20ms frames──> [ VAD gate ] ──> [ smart-turn-v2 every 160ms ]
                                                    │ P(complete) > 0.7
                              IndicConformer RNNT <─┘
                                        │
                                   vLLM (SSE)
                                        │  first clause as soon as it closes
                                   OmniVoice ──wav chunks──> mic

No fixed silence timer: the turn ends when the model says the sentence is
finished, which on Hindi separates cleanly (complete ~0.9, incomplete ~0.01)
and costs 11 ms. A 600 ms floor still applies as a backstop for the case where
the detector never fires.

Barge-in is handled by cancelling the in-flight reply the moment voiced audio
arrives while we are speaking - the caller does not have to wait us out.
"""
import asyncio, io, json, os, re, sys, time, uuid, datetime
import numpy as np, soundfile as sf, torch, aiohttp
from aiohttp import web

sys.path.insert(0, os.path.expanduser("~"))

SR = 16000
FRAME = 320                 # 20 ms
TURN_WINDOW_S = 8
TURN_EVERY_MS = 160
TURN_THRESHOLD = 0.70
RMS_GATE = 0.010
MIN_VOICED_MS = 250
HARD_STOP_MS = 2500         # backstop if the detector never fires
PAUSE_MIN_MS = 220          # only ask the detector once they actually pause
VLLM = os.environ.get("VLLM_URL", "http://localhost:8000")
TTS = os.environ.get("TTS_URL", "http://localhost:8002")
MODEL = os.environ.get("MODEL", "voais-llm")
TTS_STEPS = int(os.environ.get("TTS_STEPS", "16"))
HTML = os.path.expanduser("~/console/voice-stream.html")
REC_DIR = os.path.expanduser(os.environ.get("REC_DIR", "~/recordings"))
RECORD = os.environ.get("RECORD", "1") == "1"

SYSTEM = (
    "Aap 'Sharma & Associates' ke chartered accountant office ki receptionist hain. "
    "Aap GST, ITR, TDS, audit aur company registration jaise kaam sambhalti hain.\n\n"
    "NIYAM:\n"
    "1. Jawab HAMESHA Devanagari lipi mein. Roman ya English akshar bilkul nahi. "
    "Angrezi shabd bhi Devanagari mein: GST -> जीएसटी, ITR -> आईटीआर, PAN -> पैन.\n"
    "2. Phone par baat kar rahe hain: ek ya do chhote vakya. Koi list, koi markdown.\n"
    "3. Kaam ki baat poochhiye - kaunsa kaam, kis naam se, kab tak chahiye.\n"
    "4. Fees ya deadline ka pakka vaada mat kijiye. Kahiye ki CA saheb se "
    "confirm karke bata denge.\n"
    "5. Appointment ke liye din aur samay poochhiye."
)

GREETING_HINT = "नमस्कार, शर्मा एंड एसोसिएट्स। मैं आपकी क्या सहायता कर सकती हूँ?"

M = {}


def boot():
    import importlib.util
    from turn_model import load as load_turn
    print("loading turn detector...", flush=True)
    M["turn"], M["fe"] = load_turn("cuda")

    print("loading IndicConformer...", flush=True)
    d = os.path.expanduser("~/models/IndicConformer")
    sys.path.insert(0, d)
    spec = importlib.util.spec_from_file_location("iasr", os.path.join(d, "model_onnx.py"))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    cfg = mod.IndicASRConfig(ts_folder=d, FRAME_DURATION_MS=0.04)
    M["asr"] = mod.IndicASRModel(cfg)

    # pay both warm-ups here so no caller ever does
    warm = np.zeros(SR, dtype=np.float32)
    turn_p(warm); asr(warm, "hi")
    print("models ready", flush=True)


def turn_p(pcm: np.ndarray) -> float:
    a = pcm[-SR * TURN_WINDOW_S:]
    x = M["fe"](a, sampling_rate=SR, return_tensors="pt",
                padding="max_length", max_length=SR * TURN_WINDOW_S, truncation=True)
    x = {k: v.to("cuda") for k, v in x.items()}
    with torch.no_grad():
        return torch.sigmoid(M["turn"](**x)).item()


def asr(pcm: np.ndarray, lang="hi") -> str:
    out = M["asr"](torch.tensor(pcm).unsqueeze(0), lang, "rnnt")
    t = out[0] if isinstance(out, (list, tuple)) else out
    return str(t).strip()


async def tts_wav(sess, text):
    async with sess.post(f"{TTS}/synthesize",
                         json={"text": text, "language": "hi", "nfe": TTS_STEPS}) as r:
        if r.status != 200:
            # a silent None here made a dead TTS look like a 0 ms TTS stage
            body = (await r.text())[:120]
            print(f"TTS {r.status}: {body}", flush=True)
            return None
        return await r.read()


async def llm_stream(sess, history):
    body = {"model": MODEL, "stream": True, "temperature": 0.6, "max_tokens": 120,
            "messages": [{"role": "system", "content": SYSTEM}] + history[-8:]}
    async with sess.post(f"{VLLM}/v1/chat/completions", json=body) as r:
        buf = ""
        async for chunk in r.content.iter_any():
            buf += chunk.decode("utf-8", "ignore")
            lines = buf.split("\n"); buf = lines.pop()
            for ln in lines:
                ln = ln.strip()
                if not ln.startswith("data:"):
                    continue
                p = ln[5:].strip()
                if p == "[DONE]":
                    return
                try:
                    d = json.loads(p)["choices"][0].get("delta", {})
                    if d.get("content"):
                        yield d["content"]
                except Exception:
                    pass


class Session:
    def __init__(self, ws, sess, loop):
        self.ws, self.http, self.loop = ws, sess, loop
        self.buf = np.zeros(0, np.float32)
        self.voiced_ms = 0.0
        self.since_check = 0.0
        self.since_voice = 0.0
        self.speaking = False
        self.reply = None
        self.history = []
        self.t_speech_end = 0.0
        self.turn_no = 0
        self.dir = None
        if RECORD:
            stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            self.dir = os.path.join(REC_DIR, f"{stamp}-{uuid.uuid4().hex[:6]}")
            os.makedirs(self.dir, exist_ok=True)
            print(f"recording -> {self.dir}", flush=True)

    def log_event(self, **ev):
        if not self.dir:
            return
        ev["t_wall"] = datetime.datetime.now().isoformat(timespec="seconds")
        with open(os.path.join(self.dir, "transcript.jsonl"), "a") as f:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")

    def save_wav(self, name, pcm, sr):
        if not self.dir:
            return
        sf.write(os.path.join(self.dir, name), np.asarray(pcm, np.float32), sr)

    async def send(self, **kw):
        try: await self.ws.send_json(kw)
        except Exception: pass

    async def feed(self, pcm):
        rms = float(np.sqrt((pcm ** 2).mean()))
        ms = len(pcm) / SR * 1000
        voiced = rms > RMS_GATE

        if voiced and self.speaking:          # barge-in
            await self.cancel_reply()
            await self.send(t="barge")

        self.buf = np.concatenate([self.buf, pcm])[-SR * 20:]
        if voiced:
            self.voiced_ms += ms; self.since_voice = 0.0
        elif self.voiced_ms > 0:
            self.since_voice += ms

        if self.voiced_ms < MIN_VOICED_MS or self.reply:
            return
        # still mid-word: the detector has nothing to judge yet
        if self.since_voice < PAUSE_MIN_MS:
            self.since_check = 0.0
            return
        self.since_check += ms
        if self.since_check < TURN_EVERY_MS:
            return
        self.since_check = 0.0

        p = await self.loop.run_in_executor(None, turn_p, self.buf)
        await self.send(t="turn", p=round(p, 3))
        if p > TURN_THRESHOLD or self.since_voice > HARD_STOP_MS:
            self.t_speech_end = time.perf_counter()
            audio, self.buf = self.buf.copy(), np.zeros(0, np.float32)
            self.voiced_ms = 0.0; self.since_voice = 0.0
            self.reply = asyncio.create_task(self.respond(audio, p))

    async def cancel_reply(self):
        if self.reply and not self.reply.done():
            self.reply.cancel()
        self.speaking = False

    async def respond(self, audio, p):
        try:
            t0 = self.t_speech_end
            self.turn_no += 1
            n = self.turn_no
            self.save_wav(f"caller_{n:02d}.wav", audio, SR)
            text = await self.loop.run_in_executor(None, asr, audio, "hi")
            t_stt = time.perf_counter()
            if not text:
                self.log_event(turn=n, role="user", text="", note="asr empty")
                await self.send(t="empty"); return
            self.log_event(turn=n, role="user", text=text,
                           stt_ms=round((t_stt - t0) * 1000), turn_p=round(p, 3),
                           audio=f"caller_{n:02d}.wav")
            await self.send(t="user", m=text, stt_ms=round((t_stt - t0) * 1000), turn_p=p)
            self.history.append({"role": "user", "content": text})

            said, pend, t_first = "", "", None
            self._spoke = False   # per-reply, not per-session
            self._clip = 0
            self.speaking = True
            async for tok in llm_stream(self.http, self.history):
                if t_first is None:
                    t_first = time.perf_counter()
                    await self.send(t="ttft", ms=round((t_first - t_stt) * 1000))
                said += tok; pend += tok
                await self.send(t="delta", m=tok)
                mt = re.match(r"^([\s\S]*?[।.!?])\s", pend)
                if mt and len(mt.group(1).strip()) > 8:
                    await self.say(mt.group(1), t0)
                    pend = pend[len(mt.group(0)):]
            if pend.strip():
                await self.say(pend, t0)
            self.history.append({"role": "assistant", "content": said})
            self.log_event(turn=n, role="agent", text=said,
                           ttft_ms=round((t_first - t_stt) * 1000) if t_first else None,
                           first_audio_ms=getattr(self, "_first_ms", None))
            await self.send(t="done")
        except asyncio.CancelledError:
            await self.send(t="cancelled")
        finally:
            self.speaking = False; self.reply = None

    async def say(self, text, t0):
        wav = await tts_wav(self.http, text.strip())
        if not wav:
            await self.send(t="tts_error", m="TTS unavailable")
            return
        if not self._spoke:
            self._first_ms = round((time.perf_counter() - t0) * 1000)
            await self.send(t="first_audio", ms=self._first_ms)
            self._spoke = True
        if self.dir:
            self._clip = getattr(self, "_clip", 0) + 1
            with open(os.path.join(self.dir,
                      f"agent_{self.turn_no:02d}_{self._clip}.wav"), "wb") as f:
                f.write(wav)
        await self.ws.send_bytes(wav)


async def ws_handler(req):
    ws = web.WebSocketResponse(max_msg_size=32 * 1024 * 1024)
    await ws.prepare(req)
    s = Session(ws, req.app["s"], asyncio.get_running_loop())
    print("WS CONNECTED", req.remote, flush=True)
    await s.send(t="ready")
    async for msg in ws:
        if msg.type == aiohttp.WSMsgType.BINARY:
            pcm = np.frombuffer(msg.data, np.int16).astype(np.float32) / 32768.0
            s._n = getattr(s, "_n", 0) + len(pcm)
            if s._n // 16000 != getattr(s, "_last_s", -1):
                s._last_s = s._n // 16000
                print(f"audio {s._n/16000:.0f}s rms={float(np.sqrt((pcm**2).mean())):.4f}", flush=True)
            await s.feed(pcm)
        elif msg.type == aiohttp.WSMsgType.TEXT:
            d = json.loads(msg.data)
            if d.get("t") == "reset":
                s.history.clear(); await s.cancel_reply()
    await s.cancel_reply()
    return ws


async def index(req):
    return web.FileResponse(HTML)


async def rec_index(req):
    """Browsable list of sessions, newest first, each downloadable as a zip."""
    if not os.path.isdir(REC_DIR):
        return web.Response(text="<p>koi recording nahi</p>", content_type="text/html")
    rows = []
    for d in sorted(os.listdir(REC_DIR), reverse=True):
        full = os.path.join(REC_DIR, d)
        if not os.path.isdir(full):
            continue
        wavs = [f for f in os.listdir(full) if f.endswith(".wav")]
        turns = len([f for f in wavs if f.startswith("caller_")])
        size = sum(os.path.getsize(os.path.join(full, f)) for f in os.listdir(full))
        rows.append(f'<tr><td><a href="/recordings/{d}.zip">{d}</a></td>'
                    f'<td>{turns} turns</td><td>{len(wavs)} clips</td>'
                    f'<td>{size/1024:.0f} KB</td>'
                    f'<td><a href="/recordings/{d}/transcript.jsonl">transcript</a></td></tr>')
    body = ("<meta name=viewport content='width=device-width,initial-scale=1'>"
            "<style>body{font-family:system-ui;background:#0e1116;color:#e6e9ef;padding:16px}"
            "a{color:#35d0ba}table{border-collapse:collapse;width:100%}"
            "td{padding:8px 6px;border-bottom:1px solid #262d39;font-size:14px}</style>"
            f"<h3>Recordings ({len(rows)})</h3><table>" + "".join(rows) + "</table>")
    return web.Response(text=body, content_type="text/html")


async def rec_zip(req):
    import zipfile, tempfile
    name = req.match_info["name"]
    d = os.path.join(REC_DIR, name)
    if not os.path.isdir(d) or "/" in name or ".." in name:
        raise web.HTTPNotFound()
    tmp = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
    with zipfile.ZipFile(tmp.name, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(os.listdir(d)):
            z.write(os.path.join(d, f), arcname=f"{name}/{f}")
    return web.FileResponse(tmp.name, headers={
        "Content-Disposition": f'attachment; filename="{name}.zip"'})


async def rec_file(req):
    name, fn = req.match_info["name"], req.match_info["fn"]
    if any(("/" in x or ".." in x) for x in (name, fn)):
        raise web.HTTPNotFound()
    fp = os.path.join(REC_DIR, name, fn)
    if not os.path.isfile(fp):
        raise web.HTTPNotFound()
    return web.FileResponse(fp)


async def health(req):
    return web.json_response({"models": bool(M.get("asr")), "steps": TTS_STEPS})


async def make_app():
    app = web.Application(client_max_size=64 * 1024 * 1024)
    app["s"] = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120))
    app.router.add_get("/", index)
    app.router.add_get("/health", health)
    app.router.add_get("/recordings", rec_index)
    app.router.add_get("/recordings/{name}.zip", rec_zip)
    app.router.add_get("/recordings/{name}/{fn}", rec_file)
    app.router.add_get("/ws", ws_handler)
    return app


if __name__ == "__main__":
    boot()
    web.run_app(make_app(), host="0.0.0.0", port=7880)
