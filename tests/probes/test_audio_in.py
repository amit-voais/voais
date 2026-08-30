import base64, json, time, urllib.request, os, io
import numpy as np, soundfile as sf

WAV = os.path.expanduser("~/hindi_sample.wav")
data, sr = sf.read(WAV, dtype="float32")
if data.ndim > 1: data = data.mean(axis=1)
# resample to 16k
tgt = 16000
if sr != tgt:
    idx = np.linspace(0, len(data)-1, int(len(data)*tgt/sr))
    data = np.interp(idx, np.arange(len(data)), data).astype(np.float32)
buf = io.BytesIO()
sf.write(buf, data, tgt, format="WAV", subtype="PCM_16")
raw = buf.getvalue()
print(f"audio: {len(data)/tgt:.2f}s, {len(raw)/1024:.0f} KB, {tgt} Hz PCM16")

b64 = base64.b64encode(raw).decode()
body = {
  "model": "google/gemma-4-E2B-it",
  "messages": [{"role":"user","content":[
      {"type":"input_audio","input_audio":{"data":b64,"format":"wav"}},
      {"type":"text","text":"Transcribe this audio exactly as spoken. Then on a new line starting with REPLY:, answer it naturally in Hindi."}
  ]}],
  "max_tokens": 200, "temperature": 0.2,
}
req = urllib.request.Request("http://localhost:8000/v1/chat/completions",
        data=json.dumps(body).encode(), headers={"Content-Type":"application/json"})
t0=time.perf_counter()
try:
    r = json.load(urllib.request.urlopen(req, timeout=180))
    dt=(time.perf_counter()-t0)*1000
    print(f"\n=== GEMMA AUDIO-IN — {dt:.0f} ms ===")
    print(r["choices"][0]["message"]["content"])
    print("\nusage:", r.get("usage"))
except Exception as e:
    body_txt = getattr(e, "read", lambda: b"")()
    print("FAILED:", type(e).__name__, str(e)[:200])
    if body_txt: print("resp:", body_txt.decode()[:400])
