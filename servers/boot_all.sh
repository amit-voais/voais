#!/bin/bash
exec > "$HOME/boot.log" 2>&1
set -x
for p in $(pgrep -f 'python.*stream_server'); do kill -9 "$p" 2>/dev/null; done
for p in $(pgrep -f 'uvicorn omni_server'); do kill -9 "$p" 2>/dev/null; done
sleep 6
mkdir -p "$HOME/recordings"
cd "$HOME/tts"
OMNI_STEPS=16 setsid nohup ./.venv/bin/uvicorn omni_server:app --host 0.0.0.0 --port 8002 \
  > "$HOME/tts_server.log" 2>&1 < /dev/null &
sleep 50
echo "tts: $(curl -s --max-time 6 http://localhost:8002/health)"
setsid nohup ./.venv/bin/python "$HOME/stream_server.py" > "$HOME/stream.log" 2>&1 < /dev/null &
sleep 60
echo "stream: $(curl -s --max-time 6 http://localhost:7880/health)"
echo "recordings: $(curl -s -o /dev/null -w '%{http_code}' http://localhost:7880/recordings)"
# tunnel
for p in $(pgrep -x cloudflared); do kill -9 "$p" 2>/dev/null; done
sleep 2
setsid nohup "$HOME/cloudflared" tunnel --url http://localhost:7880 --no-autoupdate \
  > "$HOME/cf7.log" 2>&1 < /dev/null &
sleep 25
grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$HOME/cf7.log" | head -1
echo "=== ALL UP ==="
