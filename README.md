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

## Progress since this release

The code in this repository is the earlier version — the one that measures
~1.6 s end to end. Development has continued privately. The numbers below were
measured on that private build, on the same single NVIDIA L4, and are published
so the results can be checked against a real recording. **The code behind them
is not released here.**

### Measured

| | This repo | Private build |
|---|---|---|
| TTS per clip | 870 ms | **349 ms** |
| Caller stops → agent speaks | ~1600 ms | **~1050–1160 ms** |
| Cached answer | ~250 ms | **31–192 ms** |
| Language model | Gemma-4-E2B (bf16) | Gemma-4-12B (QAT int4) |
| VRAM, whole stack | — | 21.6 of 23 GB |

TTS speech quality was held constant while it got faster: an ASR round-trip on
the same sentences scored 0.988 before and 0.990 after.

### Why TTS got 2.5× faster

A profiler showed a 733 ms clip spending only 325 ms doing GPU work. The rest
was the CPU dispatching roughly 18,000 small operations to the GPU one at a
time for a single sentence. Compiling the model fused them, and the clip landed
at the 325 ms floor the profiler had predicted. Two other routes were measured
and rejected: streaming (clip cost does not fall with shorter text) and
fp8/int8 quantisation (both were *slower*, 412 and 536 ms, on a model this
small).

### Why the larger model

Sixteen questions, same prompt, one run each:

| | Knowledge outside the prompt | Answers that slipped into Roman script | Tokens/s |
|---|---|---|---|
| Gemma-4-E2B | 2/6 | 1/16 | 45.5 |
| Gemma-4-E4B int4 | 3/6 | 5/16 | 53.5 |
| **Gemma-4-12B int4** | **4/6** | **0/16** | 26.5 |

The 12B model costs ~112 ms end to end and never once left Devanagari, which is
the defect callers reported hearing.

### What made it stable

Each of these was found by recording a real call and measuring it, not by
guessing:

- **Callers cut off mid-word.** A 60 ms shortcut fired on every hesitation.
  Turns now fire from the 220 ms detector.
- **The greeting came out garbled** from cache while a fresh synthesis was
  clean. Clips had been baked from a different voice and cached under a name
  that looked right. The cache now keys on the voice the TTS server is actually
  serving.
- **No reply at all.** One caller's browser delivered speech at a tenth of the
  normal level, under the voice gate, so no turn ever fired. The server now
  normalises a quiet microphone.
- **Answering a question nobody asked.** A caller describing his company's
  services triggered the cached "what services do you offer" reply. Cached
  answers now require the caller to be asking.
- **English read as English.** When the model wrote Hinglish in Roman script
  the TTS pronounced it as English. Latin text is now transliterated before
  synthesis.

### Concurrency, load-tested

N callers at once, each asking a question the cache cannot answer, so every
turn runs the full recognition → language model → synthesis path. One run per
level. Times are measured from the moment the turn fires.

| Callers at once | All answered | Median | Slowest |
|---|---|---|---|
| 1 | 1/1 | 1158 ms | 1158 ms |
| 2 | 2/2 | 594 ms | 850 ms |
| 3 | 3/3 | 908 ms | 2270 ms |
| 4 | 4/4 | 1045 ms | 2670 ms |
| 6 | 6/6 | 2842 ms | 7622 ms |
| 8 | 8/8 | 5252 ms | 11189 ms |

Every caller got an answer at every level, but the wait grows faster than the
load. Three concurrent calls is the practical limit on one L4. Four is the
ceiling, where the slowest caller already waits 2.7 s, and beyond that it
collapses. The shape is consistent with speech synthesis queueing behind a
single worker, though that has not been isolated. Each level is one run, so
the slowest column is the worst of a handful of calls, not a percentile.

### Known limitations

- **Barge-in fails on a speakerphone.** The browser's echo canceller mutes the
  microphone while the agent speaks, so an interruption never reaches the
  server. It works with earphones.
- **Knowledge gaps remain** on questions outside the configured domain notes.
- **Three concurrent calls per GPU.** See the load test above.

### Sample call

[`samples/v7-call.mp3`](samples/v7-call.mp3) — a real 3-minute Hindi call on the
private build. Left channel is the caller, right channel is the agent.

## Licence

Apache-2.0. The models and frameworks this is built on, their licences, and the
patches made to them are in [CREDITS.md](CREDITS.md). The architecture is the
standard voice-agent pipeline — it is not original to this project.
