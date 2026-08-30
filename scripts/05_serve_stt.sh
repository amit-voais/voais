#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../services"
pip install -q fastapi uvicorn soundfile numpy onnxruntime-gpu librosa
exec uvicorn stt_server:app --host 0.0.0.0 --port 8001
