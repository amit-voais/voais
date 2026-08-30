# VoAIs — Indic Voice Agent (AI serving layer)

Region: `asia-south1` (Mumbai). This repo covers **only** the AI serving layer.
Asterisk / telephony / orchestrator / VPC are out of scope for now.

## Architecture (decided)

```
caller audio (8kHz)  ->  VAD  ->  Gemma-4-12B (audio in, text out)  ->  IndicF5 (TTS)  ->  caller
                                          |
                                  IndicConformer STT
                                  (fallback / A-B comparison only)
```

Three stages, not four. Gemma-4-12B ingests raw audio directly, so no separate
STT stage is required in the happy path.

## Model choices and why

| Stage | Model | Size | Gated |
|---|---|---|---|
| LLM (audio in) | `google/gemma-4-12B-it-qat-w4a16-ct` | 10.3 GB | no |
| TTS | `ai4bharat/IndicF5` | 1.4 GB | **yes** (auto) |
| TTS (rejected) | `ai4bharat/indic-parler-tts` | 3.6 GB | yes | 
| STT (fallback) | `ai4bharat/indic-conformer-600m-multilingual` | 2.4 GB | **yes** (auto) |

Total ~13.4 GB, fits one L4 (24 GB) with room for KV cache.
(Measured on the VM. An earlier 0.09 GB estimate for IndicConformer was wrong -
it only counted `.onnx` files; the repo ships 405 files including separate weights.)

### Why Gemma 4 over Qwen3.6-35B-A3B

Verified, not assumed:

- **Fit.** Qwen3.6-35B-A3B AWQ is 25.5 GB (QuantTrio) / 25.0 GB (cyankiwi) —
  larger than a 24 GB L4 card. Its own model card recommends `--tensor-parallel-size 4`.
  Gemma-4-12B QAT is 10.3 GB.
- **Audio input.** Gemma 4 (E2B/E4B/12B) accepts audio natively; vLLM's
  `gemma4_mm.py` wires this up (`AudioProcessorItems`, `audio_token_id`).
  Qwen3.6 has no audio model — only standalone `Qwen3-ASR`.
- **Encoder-free.** Gemma 4 12B projects raw audio waveforms straight into the
  LLM embedding space, "reducing multimodal latency" (model card). vLLM has a
  dedicated `gemma4_unified` implementation.
- **Quantization quality.** Gemma's int4 is quantization-aware trained by Google.
  The QuantTrio Qwen AWQ card states it used a "data-free quantization tool" with
  "no calibration dataset".

### TTS: IndicF5 vs indic-parler-tts

No published MOS comparison between them exists — voice quality here is a
listening decision, not a benchmark one.

| | IndicF5 | indic-parler-tts |
|---|---|---|
| Languages | 11 | **21 + English** |
| Training hours | 1,417 | **1,806** |
| Voice control | reference-audio cloning | text description (age/gender/style) |
| Size | **1.4 GB** | 3.75 GB |

Chosen: **IndicF5**, because a calling agent needs one *consistent* voice on
every call more than it needs maximum language count. Reference-audio cloning
pins the voice exactly; description-based generation can drift between calls.
Its 11 languages (Assamese, Bengali, Gujarati, Hindi, Kannada, Malayalam,
Marathi, Odia, Punjabi, Tamil, Telugu) cover most of India by speaker count.

Switch to indic-parler-tts if you need **Urdu** or a smaller regional language —
that is the main gap.

Caveat worth knowing: research finds open Indic TTS (IndicF5, indic-parler,
Chatterbox) still trails commercial systems (Sarvam Bulbul, Google Chirp3 HD) by
meaningful margins on measured phonological quality. If voice quality is the #1
product priority, a hosted commercial TTS beats both — at per-minute cost.

Run `tests/tts_listen_test.py` and judge with your own ears before committing.

### Open questions (NOT yet verified — test before trusting)

1. Gemma 4's Indic ASR accuracy vs purpose-built IndicConformer. Untested.
2. **8 kHz telephony audio.** Phone calls are narrowband; these models are
   trained on 16 kHz. Behaviour on upsampled phone audio is unknown and could
   force IndicConformer back into the happy path.
3. Indic text generation quality — neither model benchmarked for Hindi/Tamil.

`tests/asr_ab_test.py` exists to answer (1) and (2) before committing.

### Why not speech-to-speech

No open-weights S2S model outputs Indian languages. Verified: Qwen3-Omni speech
output covers 10 languages (none Indic); Moshi is English/French; Sarvam AI
publishes 14 models, none with speech output. Gemini Live API (hosted, paid)
does support Hindi native audio and is the recommended path below ~250 calls/day.

## Cost model

GPU cost per call-minute is driven by concurrency, not model choice.
At 12 h/day on one L4:

| calls/day | on-demand (Rs 69/hr) | spot (Rs 22/hr) |
|---|---|---|
| 100 | 2.76 | 0.88 |
| 500 | 0.55 | 0.18 |
| 1000 | 0.28 | 0.09 |
| 5000 | 0.06 | 0.02 |

Gemini Live is flat at ~Rs 1.07/min. Below ~250 calls/day, hosted is cheaper.

**Telephony (~Rs 0.40-0.80/min via CPaaS) dominates total cost at volume.**
The Rs 0.30-0.50/min target is a telephony negotiation, not an AI architecture problem.

## Licensing (verified)

| Model | License | Commercial use | Can the owner restrict you? |
|---|---|---|---|
| `google/gemma-4-12B-it-qat-w4a16-ct` | **Apache 2.0** | yes | no |
| `ai4bharat/IndicF5` | **MIT** | yes | no |
| `ai4bharat/indic-conformer-600m-multilingual` | **MIT** | yes | no |
| `ai4bharat/indic-parler-tts` | Apache 2.0 | yes | no |
| `k2-fsa/OmniVoice` | **NONE DECLARED** | **unclear — avoid** | n/a |

All three models in the stack are free and permissively licensed. "Gated" means
click-to-accept, not payment.

Note: Gemma **4** is Apache 2.0. This differs from Gemma 1/2/3, which shipped
under the custom "Gemma Terms of Use" containing a clause letting Google
"restrict (remotely or otherwise) usage". That clause does not apply here.
Apache 2.0 and MIT grants are perpetual and irrevocable once obtained.

Obligations are attribution-only: keep the license text and, for Apache 2.0,
any NOTICE file with your distribution. No revenue share, no usage caps, no
reporting.

OmniVoice declares no license at all. No license means no grant — do not put it
in a commercial product without written clarification from the authors.

### Data flow

Inference is entirely local. Caller audio goes:

```
caller -> your GCP VM (asia-south1) -> your response
```

Google, AI4Bharat and HuggingFace see **none** of it. The only thing they
observe is the one-time model download (HuggingFace logs the token and IP).

This is the strongest argument for self-hosting over Gemini Live: with the
hosted API, every second of customer audio is transmitted to Google. For an
Indian calling product handling personal data under the DPDP Act 2023, keeping
voice data inside your own VPC is materially simpler to defend. Confirm your
specific compliance position with a lawyer — this is an architecture note, not
legal advice.

## Live test results (2026-08-29, L4 spot in asia-south1-a)

Everything below was measured on a real VM, then torn down.

### Confirmed working

| Check | Result |
|---|---|
| vLLM 0.28.0 + Gemma 4 12B on :8000 | serving, arch `Gemma4UnifiedForConditionalGeneration` |
| Model load | 8.28 GiB VRAM, ~135 s cold start |
| Hindi text generation | "Namaste, [Company Name] mein aapka swagat hai..." in 3.0 s |
| Audio input (the reason we chose Gemma) | works |

### The 8 kHz question is answered

Real human Hindi speech (google/fleurs `hi_in`, 3 clips, ground-truth transcripts),
16 kHz vs the same audio downsampled to 8 kHz and back:

| Audio | Mean WER |
|---|---|
| 16 kHz clean | 0.184 |
| 8 kHz phone-simulated | 0.202 |
| **Degradation** | **+0.018** |

**Telephony bandwidth costs almost nothing.** The three-stage design holds;
IndicConformer does not need to return to the happy path. This was the largest
open risk in the architecture and it is now closed.

Errors are concentrated in proper nouns, which matters little for a voice agent
that needs intent rather than a verbatim transcript. Per-request latency was
1.3-1.9 s for 10-21 s clips.

Do NOT test ASR with `espeak-ng`. Synthetic formant speech is far out of
distribution and produced pure hallucination ("I am a human.") from clean input.
That result says nothing about the model. Use real speech.

### Not working yet: IndicF5 TTS

Blocked upstream, not on our side. The chain of fixes discovered:

1. `pydub`, `matplotlib`, `wandb`, `datasets` all missing -> install them
2. Loading from a local path fails: IndicF5's `model.py` calls
   `hf_hub_download(config.name_or_path, ...)`, which rejects a filesystem path.
   Load by repo id `ai4bharat/IndicF5` instead and let HF resolve from cache.
3. Real blocker: `TypeError: load_model() missing 1 required positional
   argument: 'ckpt_path'`. The `model.py` fetched from the HF cache does not
   match the bundled `f5_tts/` in the repo. Needs a version-matched f5_tts
   (or patching `model.py`) - not a dependency install.

Next attempt should pin f5_tts to the commit IndicF5 was built against, or
evaluate `ai4bharat/indic-parler-tts` instead, which is a plain transformers
model with no vendored training code.

### Environment gotchas (these cost hours)

- Image family `common-cu124-ubuntu-2204-py310` no longer exists.
  Use **`common-cu129-ubuntu-2204-nvidia-580`**.
- That image has **no pip and no conda**. `apt-get install python3-pip python3.10-venv` first.
- **A `--no-address` VM has no outbound internet.** Cloud NAT is mandatory or
  every download and apt call fails. This is a real gap in SPEC.md's
  "A1 aur A2 private only" - private still needs egress.
- vLLM needs **`ninja`** on PATH or FlashInfer JIT fails at startup.
  `VLLM_USE_FLASHINFER_SAMPLER=0` avoids the JIT entirely.
- Audio input needs **`librosa` + `soundfile`** (the `vllm[audio]` extra),
  otherwise requests fail with "Please install vllm[audio] for audio support".
- Killing vLLM by port leaves the **EngineCore worker holding VRAM**. Restarts
  then fail with "Free memory ... less than desired GPU memory utilization".
  Kill via `nvidia-smi --query-compute-apps=pid`.
- Never `pkill -f 'vllm serve'` over SSH - the pattern matches your own command
  string and kills the session.
- TTS needs its **own venv**. vLLM pulls a transformers/torch far newer than
  F5-TTS supports; sharing one venv gives
  "Tensor on device cpu is not on the expected device meta!".

### Cost actually incurred

~3.25 h on a spot L4 with NAT, about Rs 85 total (~$1). Spot was available
despite `PREEMPTIBLE_CPUS` showing 0, and was never preempted.

## Measured latency (2026-08-29, L4 spot, 6 runs/clip)

Real numbers from `tests/bench_latency.py`, not estimates.

| Stage | p50 | What it is |
|---|---|---|
| STT (Gemma audio-in) | **~800 ms** | complete clip in -> transcript out |
| **TTFT** | **78 ms** | prompt sent -> first LLM token |
| LLM total | 914 ms | first token -> last token |
| End to end | 1575 ms | audio in -> full reply text |
| **Caller waits** | **~743 ms** | before the first word of the reply exists |

Sub-second to first word. But look at the split:

    743 ms of caller wait  =  679 ms STT  +  78 ms TTFT

**STT is ~90% of it. The LLM is almost free.** Optimising the LLM would buy
nothing; the entire win is in STT. And because Gemma audio-in is not a streaming
recogniser, that ~700 ms is *dead time that begins only after the caller stops
talking*. A streaming RNNT recogniser does that work while they are still
speaking, which is why the earlier streaming analysis pointed at IndicConformer.
This measurement is the evidence for that trade, and it is worth roughly 700 ms.

TTS is not in these numbers - IndicF5 is still blocked (see above). At its
published RTF of 0.15 a 3 s reply would add ~450 ms, putting a realistic
first-audio target near 1.2 s.

### Failure mode worth knowing

One clip in six sent the model into a degenerate repetition loop: the transcript
came back as "thought thought thought..." until it hit `max_tokens=128`, and STT
took **4458 ms instead of ~800 ms** - a 6x latency blowout. The other five clips
were clean (742-885 ms, 18-22 tokens).

Production needs a guard: cap `max_tokens` for the STT call at roughly
`2 x expected_words`, and treat a completion that hits the cap as a failed turn
rather than shipping garbage to the LLM.

## IndicF5: what was actually wrong (and the trap in it)

The repo does not run as published. Four separate defects, in the order they
surface:

1. `model.py` calls `load_model(...)` without `ckpt_path`, which its own bundled
   `f5_tts` declares as a required positional argument. The weight-loading lines
   right below it are commented out. The repo was published mid-edit.
2. `load_checkpoint` passes a `torch.device` to safetensors, which wants a string.
3. The checkpoint is a flat safetensors whose keys carry `ema_model._orig_mod.`
   (the attribute name plus torch.compile's prefix) and also bundle 83 vocoder
   tensors. Nothing strips either.
4. **The trap.** transformers' `from_pretrained` re-initialises the inner CFM
   *after* `__init__` has loaded it, because those prefixed keys match nothing at
   the outer level. With `strict=False` this is completely silent: the model
   loads, runs, and emits fluent-sounding audio from **random weights**.

Point 4 is the one that matters. Every TTS sample produced before it was found
was noise. Nothing raised, nothing warned loudly enough - the only visible sign
was a `newly initialized` line buried in a wall of transformers logging.

The fix is not to load harder inside `__init__` but to inject the weights
**after** `from_pretrained` returns and then verify tensor-by-tensor:

    MATCHED 363 / MISMATCH 0     <- what a correct load looks like
    MATCHED 0   / MISMATCH 363   <- what it silently was

`services/tts_f5.py` does this and raises if fewer than 200 tensors verify.
**Never ship a `strict=False` load without a verification pass.**

### Two speed dead ends, recorded so they are not retried

- `torch.compile` (which the repo originally wrapped both models in) makes this
  **20x slower**, not faster: every new text length is a new input shape, so it
  recompiles per request. 120 s timeouts on every call. Reverted.
- `nfe_step` (the ODE-step count, the usual F5-TTS speed knob) has **no effect**
  when set after import - `infer_process` binds the default at def time. It
  must be changed at the call site or before import.

## Working system (2026-08-30)

Steady state over 3 consecutive turns, L4 spot, LLM and TTS sharing one GPU:

| Stage | p50 | was |
|---|---|---|
| STT (Gemma audio-in) | 789 ms | 694 ms |
| **First word (text)** | **886 ms** | 791 ms |
| **First audio (caller hears)** | **1410 ms** | 5649 ms |
| Full reply spoken | ~5000 ms | 14349 ms |

**4x on time-to-first-audio.** Two changes did it, and only one of them was
what I expected.

### What actually made TTS fast: nfe_step, reached properly

`nfe_step` is the ODE step count. Setting the module global after import does
nothing - `infer_process` binds it as a def-time default. It has to be passed
at the call site. Once it actually applies, the scaling is perfectly linear:

| nfe | 0.9 s of audio | 2.5 s of audio |
|---|---|---|
| 32 (upstream default) | 3747 ms | 4608 ms |
| 16 | 1876 ms | 2317 ms |
| **8 (shipped)** | **941 ms** | **1166 ms** |
| 4 | 478 ms | 582 ms |

nfe=8 is the shipped default; nfe=4 halves it again and is exposed per-request
if the quality holds up on listening.

### What I expected to matter and didn't: reference caching

`INF5Model.forward` re-runs `preprocess_ref_audio_text`, reloads the reference
WAV, and calls `.to(device)` on both models every single request. Caching all of
it at startup and calling `infer_batch_process` directly was the hypothesis for
the fixed overhead - a four-word phrase costing nearly as much as a sentence.

It was worth roughly nothing: at nfe=32 a short phrase went 3535 ms -> 3747 ms.
The flat cost was the ODE solver all along, not the preprocessing. The bypass is
kept because it is where `nfe_step` becomes reachable, but the caching itself
was not the win. Recorded so the same guess is not made twice.

### The meta-tensor trap

`from_pretrained` runs `__init__` inside a meta-device context for lazy loading.
This `__init__` builds a real Vocos vocoder and moves it to CUDA, which cannot be
done from meta - so it fails outright, and `low_cpu_mem_usage=False` does not
help. Call the constructor directly: `INF5Model(cfg)`. It already loads both the
vocoder and the CFM itself, so `from_pretrained` was contributing nothing except
the re-init that silently replaced the weights with random values.

### Where the remaining time is

    886 ms to first word  =  789 ms STT  +  97 ms TTFT

STT is still ~89% of the wait, and it is dead time: Gemma cannot begin until the
caller stops talking. A streaming RNNT recogniser (IndicConformer ships
`rnnt_decoder.onnx`) does that work while they are still speaking. That is the
next ~700 ms, and it is the last large win available.

A realistic floor for this stack:

    250 ms endpointing (unavoidable) + 50 STT + 80 TTFT + ~480 TTS at nfe=4
    ~= 860 ms to first audio

Sub-500 ms is not reachable without a genuinely streaming TTS, and no such model
exists for Indic languages today.

## Ports

| Port | Service |
|---|---|
| 8000 | vLLM — Gemma 4 (audio in, text out) |
| 8001 | IndicConformer STT (fallback) |
| 8002 | IndicF5 TTS |

## Before you run anything

1. `brew install --cask google-cloud-sdk && gcloud auth login`
2. Request quota. TWO separate quotas are required and the regional one alone
   is not enough:
   - regional `NVIDIA_L4_GPUS` in asia-south1 >= 1
   - **global `GPUS_ALL_REGIONS` >= 1** - does NOT appear in
     `gcloud compute regions describe`; read it with
     `gcloud compute project-info describe`. A limit of 0 here blocks GPU VM
     creation in every region. Approval took minutes.
3. Accept terms (one click each, instant) and create a read token at
   https://huggingface.co/settings/tokens
   - https://huggingface.co/ai4bharat/IndicF5
   - https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual
   - https://huggingface.co/ai4bharat/indic-parler-tts  (only if IndicF5 quality is poor)
4. `cp .env.example .env` and fill in
5. Set a GCP budget alert at $50 and $80

## Run order

```
scripts/00_preflight.sh        # local, read-only checks
scripts/01_gcp_provision.sh    # creates spot L4 VM + model disk
scripts/02_download_models.sh  # runs ON the VM
scripts/03_serve_llm.sh        # port 8000
scripts/05_serve_stt.sh        # port 8001
scripts/04_serve_tts.sh        # port 8002
tests/test_endpoints.sh        # curl all three
scripts/99_stop_vm.sh          # pause: stops VM, keeps models on disk
scripts/99_teardown.sh         # DELETE EVERYTHING: VM + disk + firewall
```

## Cost discipline

A running L4 costs ~Rs 20/day even idle. `scripts/99_stop_vm.sh` after every
session. A stopped VM only bills for disk (~$0.04/GB/month).
