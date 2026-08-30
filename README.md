# VoAIs — a self-hosted Indic voice agent

A real-time voice agent for Indian languages that runs entirely on one GPU.
Caller audio never leaves the machine: speech recognition, the language model
and speech synthesis are all local.

Measured end to end at **~1.6 s** on a single NVIDIA L4, of which TTS is 870 ms.
Every number below was measured, not estimated — see [FINDINGS.md](FINDINGS.md),
which also lists the assumptions that turned out to be wrong.

## How a turn works

The turn detector is the part worth looking at. It does **not** sit in the
chain — it runs alongside it, watching the same audio, and decides when the
caller has actually finished:

```
        ┌──────────────────────── mic, 20 ms frames ────────────────────────┐
        │                                                                   │
        ▼                                                                   ▼
   ┌─────────┐                                                    ┌──────────────────┐
   │ VAD     │  is anyone speaking?                               │  smart-turn-v2   │
   │ (energy)│                                                    │  "vaakya poora   │
   └────┬────┘                                                    │   hua?"  11 ms   │
        │ voiced                                                  └────────┬─────────┘
        │                                                    runs only at a pause,
        ▼                                                    every 160 ms, in parallel
   ┌─────────────────────────────────────────┐                        │
   │  rolling audio buffer                   │◄───────────────────────┘
   └────────────────┬────────────────────────┘         P(complete) > 0.70
                    │
                    ▼
   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
   │ IndicConf.   │───►│ Gemma-4-E2B  │───►│ OmniVoice    │───►│  speaker     │
   │ RNNT   ~500ms│    │ vLLM    41ms │    │ TTS    870ms │    │              │
   └──────────────┘    └──────┬───────┘    └──────▲───────┘    └──────────────┘
                              │                   │
                              └───────────────────┘
                        first clause is sent to TTS as soon as it closes,
                        the rest is generated while that clause plays

   barge-in: voiced audio while speaking cancels the in-flight reply
```

Why parallel matters: a fixed silence timer has to choose between cutting people
off and being slow. The detector reads the audio instead — 0.97 on
*"मुझे कल सुबह दस बजे अपॉइंटमेंट चाहिए।"* and 0.009 on *"मुझे कल सुबह"* — so it can
answer immediately after a finished sentence and keep waiting through a pause.

It must only be asked at a pause. Polled during speech it fires on stray
confidence and cuts the caller off after one word.

## Measured

| Stage | ms | Model |
|---|---|---|
| Turn detection | **11** | pipecat-ai/smart-turn-v2 |
| STT | **~500** | ai4bharat/indic-conformer-600m (RNNT, RTF 0.045) |
| LLM first token | **41** | google/gemma-4-E2B-it on vLLM |
| TTS | **870** | k2-fsa/OmniVoice, num_step=16 |
| **End to end** | **~1600** | |

Cached opening phrases answer in ~250 ms because the audio already exists.

Roughly 5 concurrent calls fit on one L4 — derived from per-turn GPU time, not
load-tested. TTS is 55% of it.

## Run it

Needs one NVIDIA GPU with ≥24 GB and a HuggingFace token — `IndicConformer`
and `IndicF5` are gated, so accept their terms on HuggingFace first, then
`hf auth login`.

```bash
hf download google/gemma-4-E2B-it
hf download k2-fsa/OmniVoice
hf download pipecat-ai/smart-turn-v2                    --local-dir ~/models/smart-turn-v2
hf download ai4bharat/indic-conformer-600m-multilingual --local-dir ~/models/IndicConformer
```

```bash
vllm serve google/gemma-4-E2B-it --served-model-name voais-llm \
  --port 8000 --max-model-len 4096 --dtype bfloat16 \
  --gpu-memory-utilization 0.55 --enable-prefix-caching
```

```bash
OMNI_STEPS=16 uvicorn omni_server:app --host 0.0.0.0 --port 8002
python servers/stream_server.py          # serves the UI and /ws on :7880
```

`pip install "vllm[audio]"` is required or audio requests fail.

## Before you trust a change

Synthesize a known sentence, read it back with the ASR, compare. That test
(`tests/probes/q2.py`) caught three separate TTS defects that all sounded
plausible while being wrong — including one where the audio was noise the
recogniser scored at 0.34.

```bash
python tests/probes/q2.py         # TTS quality by ASR round-trip
python tests/probes/ws_probe.py   # end-to-end: must print first_audio + AUDIO
```

`/health` returning 200 is not enough. It checks that models loaded, not that a
call works — a broken `Session.__init__` once passed health and failed every
real connection.

## Voice

The TTS clones voice, pace **and loudness** from one reference clip. Supply your
own: 5–10 s of Hindi, spoken rather than read, cut at silence, RMS ≈ -18 dBFS.
All four voice defects we hit came from one bad reference. See `refs/README.md`.

## Recording

Every call is written to `~/recordings/<session>/` — caller and agent as
separate wav files, plus a `transcript.jsonl` with per-turn timings. Separate
rather than mixed because a full-duplex model finetuned later needs the two
sides apart.

Browse and download at `/recordings`.

## Licence

Apache-2.0. The models and frameworks this is built on, their licences, and the
patches made to them are in [CREDITS.md](CREDITS.md). The architecture is the
standard voice-agent pipeline — it is not original to this project.
