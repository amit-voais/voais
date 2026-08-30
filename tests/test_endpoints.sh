#!/usr/bin/env bash
# Curl all three endpoints. Run on the VM, or locally after port-forwarding:
#   gcloud compute ssh $VM_NAME --zone $ZONE --tunnel-through-iap -- \
#     -L 8000:localhost:8000 -L 8001:localhost:8001 -L 8002:localhost:8002
set -uo pipefail
H=${HOST:-localhost}

echo "=== :8000 LLM (Gemma 4) — text ==="
curl -sS "http://$H:8000/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"voais-llm","messages":[{"role":"user","content":"Namaste, aap kaise hain? Ek line mein jawab dein."}],"max_tokens":64}' \
  | python3 -m json.tool 2>/dev/null | head -30 || echo "  FAILED"

echo
echo "=== :8000 LLM — audio in (the whole point of Gemma 4) ==="
if [ -f data/samples/hi_sample.wav ]; then
  B64=$(base64 < data/samples/hi_sample.wav | tr -d '\n')
  curl -sS "http://$H:8000/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"voais-llm\",\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"audio_url\",\"audio_url\":{\"url\":\"data:audio/wav;base64,$B64\"}},{\"type\":\"text\",\"text\":\"Transcribe this audio.\"}]}],\"max_tokens\":128}" \
    | python3 -m json.tool 2>/dev/null | head -30 || echo "  FAILED"
else
  echo "  SKIP — put a 16kHz mono WAV at data/samples/hi_sample.wav"
fi

echo
echo "=== :8001 STT (IndicConformer) ==="
curl -sS "http://$H:8001/health" | python3 -m json.tool 2>/dev/null || echo "  FAILED"
if [ -f data/samples/hi_sample.wav ]; then
  curl -sS -X POST "http://$H:8001/transcribe" \
    -F "file=@data/samples/hi_sample.wav" -F "language=hi" \
    | python3 -m json.tool 2>/dev/null || echo "  FAILED"
fi

echo
echo "=== :8002 TTS (IndicF5) ==="
curl -sS "http://$H:8002/health" | python3 -m json.tool 2>/dev/null || echo "  FAILED"
curl -sS -X POST "http://$H:8002/synthesize" \
  -H 'Content-Type: application/json' \
  -d '{"text":"Namaste, main aapki kya madad kar sakta hoon?","language":"hi"}' \
  -o /tmp/tts_out.wav \
  && echo "  wrote /tmp/tts_out.wav ($(du -h /tmp/tts_out.wav | cut -f1))" || echo "  FAILED"
