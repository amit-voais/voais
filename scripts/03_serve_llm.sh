#!/usr/bin/env bash
# vLLM serving Gemma-4-12B (audio in, text out) on :8000
# gpu-memory-utilization is 0.70, NOT 0.92: TTS shares this GPU.
#   0.70 * 24GB = ~16.8GB for vLLM (10.3 weights + ~6.5 KV cache)
#   leaves ~7GB for IndicF5 TTS (1.4GB) + CUDA context + headroom
set -euo pipefail
MNT=${MNT:-/mnt/models}

pip install -q "vllm>=0.28.0"

exec vllm serve "$MNT/gemma-4-12B-it-qat-w4a16-ct" \
  --served-model-name voais-llm \
  --host 0.0.0.0 --port 8000 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.70 \
  --enable-prefix-caching \
  --limit-mm-per-prompt '{"audio":1}' \
  --max-num-seqs 32
