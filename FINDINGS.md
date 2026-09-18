# VoAIs — what was actually built and measured (2026-08-30)

Everything below was measured on the VM, not estimated. Where a number is an
estimate it says so. Where an earlier claim turned out wrong, the wrong claim is
left in with the correction, because the wrong ones are the useful part.

## Running system

```
voais-gpu-2 | g2-standard-12 | 1x NVIDIA L4 24GB | asia-south1-b
project hidrogen-500914 | billing 010D07-FF0401-07246F (iiml-credit, INR)
```

| Port | Service | Model |
|------|---------|-------|
| 8000 | vLLM | google/gemma-4-E2B-it (bf16), served as `voais-llm` |
| 8002 | TTS | k2-fsa/OmniVoice, num_step=16 |
| 7880 | streaming agent | IndicConformer RNNT + smart-turn-v2 |

Restart: `bash ~/apply_sil.sh` brings up TTS then the agent (~100 s).
Public URL: `~/cloudflared tunnel --url http://localhost:7880`

## Measured latency, per stage

| Stage | ms | Notes |
|---|---|---|
| turn detection | **11** | smart-turn-v2, after warm-up |
| STT (IndicConformer RNNT) | **548** | 12.24 s of audio, RTF 0.045 |
| STT, short utterance | **198** | as seen in a live turn |
| LLM TTFT | **39–107** | Gemma-4-E2B bf16 on vLLM |
| TTS (OmniVoice step=16) | **866** | ~3.2 s of audio out |
| End-to-end, live | **~1100** | plus ~100-200 ms Cloudflare in testing |

Cached greeting is 0 ms and hides most of the TTS cost on the first turn.

## Turn detector

`pipecat-ai/smart-turn-v2`, 362 MB, Wav2Vec2 + attention pooling.

| Utterance | P(complete) |
|---|---|
| मुझे कल सुबह दस बजे अपॉइंटमेंट चाहिए। | 0.974 |
| जी हाँ, मेरा नाम अमित है। | 0.859 |
| मुझे कल सुबह | 0.009 |
| जी हाँ, मेरा नाम | 0.009 |

Clean separation on Hindi even though the model is trained mostly on English -
it reads prosody, not words, and that transfers. 11 ms per call.

**It must only be run at a pause.** Polling it during speech asks a question it
was not trained on, and a stray high score cuts the caller off after one word.
`PAUSE_MIN_MS = 220` in stream_server.py.

## The method that caught every TTS bug

Synthesize a known sentence, run the result back through the ASR, compare to the
original. No listening required, and it catches failures that sound plausible.

```
match = SequenceMatcher(target, asr(tts(target))).ratio()
```

Three bugs, all found this way, all mine:

| Symptom | Cause | Evidence | Fix |
|---|---|---|---|
| "ulta sidha bol raha hai" | num_step=4 | match 0.34, ASR heard "यहाँ तिरते बत्तीस की ओर" | step 16 -> match 0.99 |
| "bahut slow" | 12 s reference; the clone copies its pace | 5.44 s out vs ~2.5 s human | 5 s slice -> 3.0 s |
| "करता है" spoken before every reply | reference ended mid-phrase; OmniVoice continues from it | ASR output did not start with नमस्ते | cut reference at a silence gap |

`num_step` quality cliff (same sentence):

| steps | ms | match |
|---|---|---|
| 2 | 306 | 0.00 |
| 4 | 362 | 0.34 |
| 8 | 513 | 0.00 |
| **16** | **870** | **0.99** |
| 32 | 1592 | 0.96 |

Below 16 the audio is noise that still sounds like speech. `speed=1.6+` also
destroys it (match 0.00) - that parameter is not usable.

## Claims I got wrong

| Claimed | Actual |
|---|---|
| Human-1 bf16 fits an L4 because moshi defaults to bf16 | OOM - the loader materialises fp32 on GPU first. Had to convert the checkpoint on disk (30.75 -> 15.38 GB) |
| Human-1 frame budget 52/80 ms, 65% used | 78/80 ms, ~97% used |
| Human-1 concurrency ~10-25 | **1** - and for a software reason: `asyncio.Lock` + `batch_size=1` in Kyutai's demo server, not a hardware limit |
| IndicF5 TTS ~120 ms | never measured; OmniVoice at usable quality is 866 ms |
| Cloudflare adds ~400 ms | its POP is New Delhi; nearer 100-200 ms |
| smart-turn loads via AutoModelForAudioClassification | it silently attaches a random head. Real arch is pool_attention + 4-layer classifier; weights now verified 12/12 |

## GCP constraints (verified by trying)

Every modern GPU family is 0 on this billing account:
`RTX_PRO_6000, B200, H100, H100_MEGA, H200`. Only L4/T4/V100/P100/P4 have any.

- G4 quota request: denied twice, in ~2 seconds (automated)
- L4 1 -> 4 request: denied
- `g4-standard-48`: `Quota 'GPUS_PER_GPU_FAMILY' exceeded`
- `g4-standard-6/24`, `a3-highgpu-1g`: `ZONE_RESOURCE_POOL_EXHAUSTED` in
  asia-south1-c, asia-south2-a/c and us-central1-b

Both walls are up: no quota **and** no capacity. Stopping the L4 does not help -
quota is per family, and G4 was tested with zero GPUs running.

Also: stopping a GPU VM releases its capacity. `voais-gpu-1` could not be
restarted (STOCKOUT) and had to be rebuilt in asia-south1-b.

## TTS engines, measured the same way

| engine | ms | out sec | ASR match | leading junk |
|---|---|---|---|---|
| OmniVoice step=16 | **847-903** | 3.2-4.2 | **0.96-0.97** | none |
| OmniVoice step=24 | 1175-1238 | 3.2-4.2 | 0.96-0.97 | none |
| IndicF5 | - | - | - | **would not load** |
| indic-parler-tts (blocking) | 5132-6802 | 2.5-3.5 | 0.81-0.88 | drops words |
| indic-parler-tts (streaming) | **1966 first chunk** | 3.0 | 0.81-0.88 | drops words |

step=24 costs ~350 ms more and buys nothing. **16 is the setting.**

### IndicF5: five defects, still does not run

The README in this repo listed four; a fifth surfaced.

1. `model.py` calls `load_model()` without `ckpt_path` - patched
2. `load_checkpoint` hands safetensors a `torch.device` - patched
3. keys carry `ema_model._orig_mod.` - patched
4. checkpoint bundles 83 vocoder tensors that break a strict load - patched
5. `self.ema_model = torch.compile(load_model(...))` renames every live
   parameter with `_orig_mod.`, so the injected weights match nothing.
   `tts_fast.py` catches this: `weight verification failed: only 0 tensors match`

That verification is the thing worth keeping. Without it the model would have
loaded, run, and produced fluent-sounding audio from random weights - the exact
trap this repo's README documents.

Not pursued further. The earlier conclusion in the README was right: pin
`f5_tts` to the commit IndicF5 was built against, or use a model that does not
vendor its own training code.

indic-parler-tts was measured after the above and lost on both axes: 6-8x
slower and a lower match, dropping words outright ("जी बिल्कुल, कल सुबह" came
back as "ई बुक हो सुबह"). It does speak at a more natural pace - 2.50 s where
OmniVoice takes 3.24 s for the same sentence - but 5-7 s of generation is not
usable on a call. Being autoregressive it can stream within an utterance, so that was measured
separately: first chunk at **1966 ms** (play_steps 0.5 s) or 1996 ms (1.0 s),
against OmniVoice's 870 ms for the *complete* clip. It loses in both modes.

Installing parler-tts pins `transformers` back to 4.46 and breaks OmniVoice's
import - the TTS server then answers 503 and, until this was fixed,
`stream_server.say()` swallowed that silently and the UI showed a TTS stage of
0 ms and a turn total of 549 ms that was pure fiction. Parler now lives in its
own venv (`~/.venv-parler`) and `tts_wav` logs non-200 responses.

**OmniVoice at num_step=16 is the choice.**

## Open
2. Cached phrases are not wired into stream_server.py. They exist in
   agent_experiment/bot.py and would hide most of the 866 ms.
3. Streaming STT: IndicConformer is RNNT and can emit partials, but the current
   server still transcribes the whole utterance after the turn ends.
4. Gemma's Indic ASR was never A/B'd against IndicConformer. IndicConformer is
   2x faster (548 vs 1120 ms) so it won on speed alone.
5. Nothing here has been tested on 8 kHz telephony audio.
