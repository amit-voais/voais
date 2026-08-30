"""Gemma-4-12B audio-in vs IndicConformer, on 16 kHz AND 8 kHz telephony audio.

This exists because two things were never verified before choosing Gemma:
  1. Gemma's Indic ASR accuracy vs a purpose-built Indic model
  2. how either behaves on 8 kHz narrowband phone audio

Both models are trained on 16 kHz. Real calls are 8 kHz. If Gemma degrades badly
on downsampled audio, IndicConformer goes back into the happy path and the
pipeline becomes four stages again.

Usage:
    python tests/asr_ab_test.py --manifest data/samples/manifest.jsonl

manifest.jsonl, one JSON object per line:
    {"audio": "data/samples/hi_01.wav", "language": "hi", "text": "ground truth"}
"""
import argparse
import base64
import json
import re
import unicodedata

import requests

LLM = "http://localhost:8000/v1/chat/completions"
STT = "http://localhost:8001/transcribe"


def normalize(s: str) -> str:
    s = unicodedata.normalize("NFKC", s.strip().lower())
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", s))


def wer(ref: str, hyp: str) -> float:
    r, h = normalize(ref).split(), normalize(hyp).split()
    if not r:
        return 0.0 if not h else 1.0
    d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
    return d[len(r)][len(h)] / len(r)


def to_8k_and_back(path: str) -> bytes:
    """Simulate a phone call: 16k -> 8k -> 16k. Detail lost is not recovered."""
    import io

    import librosa
    import soundfile as sf

    y, _ = librosa.load(path, sr=16000, mono=True)
    y = librosa.resample(librosa.resample(y, orig_sr=16000, target_sr=8000),
                         orig_sr=8000, target_sr=16000)
    buf = io.BytesIO()
    sf.write(buf, y, 16000, format="WAV")
    return buf.getvalue()


def ask_gemma(raw: bytes) -> str:
    b64 = base64.b64encode(raw).decode()
    body = {
        "model": "voais-llm",
        "messages": [{"role": "user", "content": [
            {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{b64}"}},
            {"type": "text", "text": "Transcribe this audio verbatim. Output only the transcript."},
        ]}],
        "max_tokens": 256, "temperature": 0.0,
    }
    r = requests.post(LLM, json=body, timeout=120)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def ask_conformer(raw: bytes, lang: str) -> str:
    r = requests.post(STT, files={"file": ("a.wav", raw, "audio/wav")},
                      data={"language": lang}, timeout=120)
    r.raise_for_status()
    return r.json()["text"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    args = ap.parse_args()

    rows = [json.loads(x) for x in open(args.manifest) if x.strip()]
    acc: dict[str, list[float]] = {k: [] for k in
                                   ("gemma_16k", "conformer_16k", "gemma_8k", "conformer_8k")}

    for item in rows:
        ref, lang = item["text"], item.get("language", "hi")
        wav16 = open(item["audio"], "rb").read()
        wav8 = to_8k_and_back(item["audio"])
        for key, fn, raw in (
            ("gemma_16k", lambda b: ask_gemma(b), wav16),
            ("conformer_16k", lambda b: ask_conformer(b, lang), wav16),
            ("gemma_8k", lambda b: ask_gemma(b), wav8),
            ("conformer_8k", lambda b: ask_conformer(b, lang), wav8),
        ):
            try:
                acc[key].append(wer(ref, fn(raw)))
            except Exception as exc:
                print(f"  {key} failed on {item['audio']}: {exc}")

    print(f"\n{'variant':<18}{'mean WER':>10}{'n':>6}   (lower is better)")
    print("-" * 40)
    for k, v in acc.items():
        if v:
            print(f"{k:<18}{sum(v) / len(v):>10.3f}{len(v):>6}")

    g16, g8 = acc["gemma_16k"], acc["gemma_8k"]
    if g16 and g8:
        drop = (sum(g8) / len(g8)) - (sum(g16) / len(g16))
        print(f"\nGemma degradation from 8 kHz: {drop:+.3f} WER")
        if drop > 0.10:
            print("VERDICT: Gemma degrades badly on phone audio.")
            print("         Put IndicConformer back in the happy path (4 stages).")
        else:
            print("VERDICT: Gemma holds up on phone audio. Keep the 3-stage design.")


if __name__ == "__main__":
    main()
