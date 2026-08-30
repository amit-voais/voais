#!/bin/bash
cd "$HOME/tts"
exec ./.venv/bin/python "$HOME/stream_server.py"
