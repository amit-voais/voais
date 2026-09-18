"""Gemma-4-12B as a Pipecat STT service.

Gemma ingests audio directly, so this is not a wrapper around a separate
recogniser - the same model that answers also transcribes. It speaks the OpenAI
chat API through vLLM, so audio goes in as a base64 data URI.

Two things this guards against, both learned the hard way on real audio:

* a completion that runs to `max_tokens` is a degenerate repetition loop, not a
  transcript ("thought thought thought..."). Roughly one clip in six triggered
  it, and it cost 4.5 s instead of 0.8 s. Those turns are dropped rather than
  passed downstream.
* Gemma is not a streaming recogniser. It cannot start until the caller stops
  talking, which is why this stage dominates turn latency. Swapping in a
  streaming RNNT is the next structural win; the interface here does not change.
"""
import base64
import io
import time
import wave
from collections.abc import AsyncGenerator

import aiohttp
from loguru import logger
from pipecat.frames.frames import ErrorFrame, Frame, TranscriptionFrame
from pipecat.services.stt_service import STTService
from pipecat.utils.time import time_now_iso8601

PROMPT = "Transcribe this audio verbatim in Devanagari. Output only the transcript."


class GemmaSTTService(STTService):
    def __init__(
        self,
        *,
        base_url: str = "http://localhost:8000",
        model: str = "voais-llm",
        max_tokens: int = 96,
        sample_rate: int = 16000,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)
        self._url = base_url.rstrip("/") + "/v1/chat/completions"
        self._model = model
        self._max_tokens = max_tokens
        self._session: aiohttp.ClientSession | None = None

    async def start(self, frame):
        await super().start(frame)
        if self._session is None:
            self._session = aiohttp.ClientSession()

    async def stop(self, frame):
        await super().stop(frame)
        if self._session:
            await self._session.close()
            self._session = None

    def _wav(self, pcm: bytes) -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(self.sample_rate)
            w.writeframes(pcm)
        return buf.getvalue()

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        if not audio:
            return
        if self._session is None:
            self._session = aiohttp.ClientSession()

        b64 = base64.b64encode(self._wav(audio)).decode()
        body = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "temperature": 0.0,
            "messages": [{"role": "user", "content": [
                {"type": "audio_url",
                 "audio_url": {"url": f"data:audio/wav;base64,{b64}"}},
                {"type": "text", "text": PROMPT},
            ]}],
        }

        await self.start_ttfb_metrics()
        t0 = time.perf_counter()
        try:
            async with self._session.post(self._url, json=body) as r:
                if r.status != 200:
                    yield ErrorFrame(f"stt http {r.status}: {(await r.text())[:200]}")
                    return
                data = await r.json()
        except Exception as exc:
            yield ErrorFrame(f"stt failed: {exc}")
            return
        await self.stop_ttfb_metrics()

        used = data.get("usage", {}).get("completion_tokens", 0)
        if used >= self._max_tokens:
            logger.warning(f"stt hit the token cap ({used}) - repetition loop, turn dropped")
            return

        text = data["choices"][0]["message"]["content"].strip()
        if not text:
            return
        logger.debug(f"stt {(time.perf_counter()-t0)*1000:.0f}ms: {text[:60]}")
        yield TranscriptionFrame(text, "", time_now_iso8601(), finalized=True)
