"""Measure the latency that actually matters for a voice agent.

Interactive clicking gives you one number with your own reaction time baked in.
This runs a fixed set of conversational-length clips N times and reports the
distribution, separating the stages so you can see where the time really goes.

Reported per stage:
    stt    audio bytes -> transcript          (Gemma audio-in, non-streaming)
    ttft   prompt sent -> first token         <- the number a caller feels
    llm    first token -> last token
    e2e    audio in -> full reply text

TTS is measured separately because IndicF5 is non-autoregressive: it cannot
stream, so its cost is a fixed block per utterance, not a per-token cost.

Usage (on the VM):
    python tests/bench_latency.py --runs 5
"""
import argparse
import base64
import json
import statistics as st
import time
import urllib.request

VLLM = "http://localhost:8000"
SYSTEM = ("Aap ek Indian call center agent hain. Hindi mein chhota, natural jawab "
          "dijiye. Ek ya do vakya se zyada nahi.")


def _post(path, body, stream=False):
    req = urllib.request.Request(VLLM + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=180)


def stt(wav_path):
    b64 = base64.b64encode(open(wav_path, "rb").read()).decode()
    body = {"model": "voais-llm", "max_tokens": 128, "temperature": 0.0,
            "messages": [{"role": "user", "content": [
                {"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64," + b64}},
                {"type": "text", "text": "Transcribe this audio verbatim. Output only the transcript."}]}]}
    t = time.perf_counter()
    r = json.load(_post("/v1/chat/completions", body))
    return r["choices"][0]["message"]["content"].strip(), (time.perf_counter() - t) * 1000


def llm_stream(prompt):
    body = {"model": "voais-llm", "stream": True, "max_tokens": 150, "temperature": 0.3,
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": prompt}]}
    t = time.perf_counter()
    ttft, parts = None, []
    resp = _post("/v1/chat/completions", body, stream=True)
    for raw in resp:
        line = raw.decode().strip()
        if not line.startswith("data: "):
            continue
        payload = line[6:]
        if payload == "[DONE]":
            break
        d = json.loads(payload)["choices"][0]["delta"].get("content")
        if not d:
            continue
        if ttft is None:
            ttft = (time.perf_counter() - t) * 1000
        parts.append(d)
    return "".join(parts), ttft, (time.perf_counter() - t) * 1000


def summarize(name, xs, unit="ms"):
    if not xs:
        print(f"  {name:<10} no data")
        return
    xs = sorted(xs)
    p95 = xs[min(len(xs) - 1, int(round(0.95 * (len(xs) - 1))))]
    print(f"  {name:<10} mean {st.mean(xs):7.0f}  p50 {st.median(xs):7.0f}  "
          f"p95 {p95:7.0f}  min {xs[0]:7.0f}  max {xs[-1]:7.0f}  {unit}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--clips", nargs="*", default=[
        "/mnt/models/samples/turn_short.wav",
        "/mnt/models/samples/turn_med.wav",
    ])
    args = ap.parse_args()

    print("warming up (first call pays compile/cache costs)...")
    try:
        llm_stream("Namaste")
    except Exception as e:
        print("  warmup failed:", e)

    for clip in args.clips:
        stts, ttfts, llms, e2es = [], [], [], []
        print(f"\n=== {clip.split('/')[-1]} x{args.runs} ===")
        for i in range(args.runs):
            try:
                t0 = time.perf_counter()
                text, s_ms = stt(clip)
                reply, ttft, l_ms = llm_stream(text)
                e2e = (time.perf_counter() - t0) * 1000
                stts.append(s_ms); ttfts.append(ttft); llms.append(l_ms); e2es.append(e2e)
                if i == 0:
                    print(f"  heard : {text[:70]}")
                    print(f"  reply : {reply[:70]}")
            except Exception as e:
                print(f"  run {i} failed: {str(e)[:120]}")
        print()
        summarize("stt", stts)
        summarize("ttft", [x for x in ttfts if x is not None])
        summarize("llm", llms)
        summarize("e2e", e2es)
        if ttfts and e2es:
            print(f"\n  -> caller waits ~{st.median([a+b for a,b in zip(stts,[t or 0 for t in ttfts])]):.0f} ms "
                  f"before the first word of the reply exists")


if __name__ == "__main__":
    main()
