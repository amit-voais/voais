# Credits

This project is assembled from open-source models and frameworks. The
architecture — VAD, a turn-detection model, streaming STT → LLM → chunked TTS,
with barge-in — is the standard voice-agent pipeline, not something invented
here. What is ours is the specific Indic stack, the patches listed below, and
the measurements in `FINDINGS.md`.

Every component below is under a permissive licence that allows commercial use.

## Models

| Component | Project | Licence |
|---|---|---|
| Turn detection | [pipecat-ai/smart-turn-v2](https://huggingface.co/pipecat-ai/smart-turn-v2) | BSD-2-Clause |
| STT | [ai4bharat/indic-conformer-600m-multilingual](https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual) | MIT |
| LLM | [google/gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it) | Apache-2.0 |
| TTS | [k2-fsa/OmniVoice](https://huggingface.co/k2-fsa/OmniVoice) | Apache-2.0 |
| TTS (evaluated) | [ai4bharat/IndicF5](https://huggingface.co/ai4bharat/IndicF5) | MIT |
| TTS (evaluated) | [ai4bharat/indic-parler-tts](https://huggingface.co/ai4bharat/indic-parler-tts) | Apache-2.0 |
| S2S (evaluated) | [VoiceArena/Human-1](https://huggingface.co/VoiceArena/Human-1) | CC-BY-4.0 |

## Frameworks

| Project | Licence |
|---|---|
| [pipecat-ai/pipecat](https://github.com/pipecat-ai/pipecat) | BSD-2-Clause |
| [vLLM](https://github.com/vllm-project/vllm) | Apache-2.0 |
| [kyutai-labs/moshi](https://github.com/kyutai-labs/moshi) | Apache-2.0 |

## Reference audio

The default voice reference is derived from [google/fleurs](https://huggingface.co/datasets/google/fleurs)
(CC-BY-4.0), Hindi validation split. It is a placeholder — replace it with your
own recording before shipping.

## Changes we made (required by Apache-2.0 and CC-BY-4.0)

**IndicF5** — the published repo does not run. Patched in `f5_tts/infer/utils_infer.py`
and `model.py`:
1. `model.py` called `load_model()` without the required `ckpt_path`
2. `load_checkpoint` passed a `torch.device` where safetensors wants a string
3. checkpoint keys carry an `ema_model._orig_mod.` prefix that nothing stripped
4. the checkpoint bundles 83 vocoder tensors that break a strict load
5. `torch.compile` renames live parameters, so injected weights match nothing —
   this one is unresolved and IndicF5 is not in the shipped pipeline

**smart-turn-v2** — `AutoModelForAudioClassification` silently attaches a
randomly-initialised head to this checkpoint. `servers/turn_model.py` implements
the real architecture (attention pooling + 4-layer classifier) and verifies all
12 head tensors on load rather than trusting `strict=False`.

**OmniVoice** — wrapped in `servers/omni_server.py` behind IndicF5's HTTP
contract. `num_step` lowered from 32 to 16 (below 16 the output is noise that
still sounds like speech — see FINDINGS.md), and output is RMS-normalised,
which the upstream example does not do.

**Human-1** — evaluated only. Converted fp32 → bf16 on disk to fit a 24 GB L4;
the loader materialises fp32 on GPU first and OOMs otherwise.

## Thanks

To AI4Bharat for the Indic speech models, to the Pipecat team for smart-turn-v2
and for a voice-agent framework that already had barge-in and cached phrases
solved, to k2-fsa for OmniVoice, and to Josh Talks / VoiceArena for publishing
Human-1 along with an honest evaluation of its limits — that honesty saved us
from shipping it.
