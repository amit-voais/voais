"""IndicF5 as a Pipecat TTS service, with a cached-phrase fast path.

IndicF5 is a flow-matching model: it is non-autoregressive, so it cannot stream
within an utterance. Cost is dominated by a fixed term that scales with the ODE
step count and not with text length - measured on an L4:

    nfe   fixed    per second of audio
      4   419 ms    65 ms
      8   814 ms   141 ms
     16  1627 ms   276 ms

So shortening the text barely helps; lowering `nfe` is the real lever. nfe=8 is
the default here as the point where quality still holds up on a phone.

The cache exists because that fixed cost cannot be optimised away. A call centre
agent opens every call with the same sentence, and greetings are synthesised
once at startup and replayed at zero latency while the real reply is still being
generated behind them.

One hard-won detail: the text must be Devanagari. Feeding romanised Hindi
("Namaste, main aapki madad...") produces confident gibberish at roughly half the
correct duration, because the model has no idea how to pronounce Latin script.
Callers hear it as a broken robot. `looks_devanagari()` is exported so the
pipeline can assert on it.
"""
import io
import re
import wave
from collections.abc import AsyncGenerator

import aiohttp
from loguru import logger
from pipecat.frames.frames import ErrorFrame, Frame, TTSAudioRawFrame, TTSStartedFrame, TTSStoppedFrame
from pipecat.services.tts_service import TTSService

DEVANAGARI = re.compile(r"[ऀ-ॿ]")
LATIN = re.compile(r"[A-Za-z]")


def looks_devanagari(text: str) -> bool:
    """True when the text is mostly Devanagari - the only form this model speaks."""
    return len(DEVANAGARI.findall(text)) > len(LATIN.findall(text))


class IndicF5TTSService(TTSService):
    def __init__(
        self,
        *,
        base_url: str = "http://localhost:8002",
        nfe: int = 8,
        sample_rate: int = 24000,
        cache_phrases: tuple[str, ...] = (),
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)
        self._url = base_url.rstrip("/") + "/synthesize"
        self._nfe = nfe
        self._cache_phrases = cache_phrases
        self._cache: dict[str, bytes] = {}
        self._session: aiohttp.ClientSession | None = None

    async def start(self, frame):
        await super().start(frame)
        if self._session is None:
            self._session = aiohttp.ClientSession()
        for phrase in self._cache_phrases:
            try:
                self._cache[phrase] = await self._synth(phrase)
                logger.info(f"tts cached: {phrase[:40]}")
            except Exception as exc:
                logger.warning(f"tts could not cache {phrase[:20]}: {exc}")

    async def stop(self, frame):
        await super().stop(frame)
        if self._session:
            await self._session.close()
            self._session = None

    @staticmethod
    def _pcm(wav_bytes: bytes) -> bytes:
        with wave.open(io.BytesIO(wav_bytes), "rb") as w:
            return w.readframes(w.getnframes())

    async def _synth(self, text: str) -> bytes:
        if self._session is None:
            self._session = aiohttp.ClientSession()
        async with self._session.post(
            self._url, json={"text": text, "language": "hi", "nfe": self._nfe}
        ) as r:
            if r.status != 200:
                raise RuntimeError(f"tts http {r.status}: {(await r.text())[:160]}")
            return self._pcm(await r.read())

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        text = text.strip()
        if not text:
            return
        if not looks_devanagari(text):
            # Not fatal - say it anyway - but this is the failure that sounds like
            # a broken model rather than a broken prompt, so make it loud.
            logger.warning(f"tts got non-Devanagari text, output will be wrong: {text[:60]}")

        cached = self._cache.get(text)
        yield TTSStartedFrame()
        try:
            if cached is not None:
                logger.debug(f"tts cache hit: {text[:40]}")
                pcm = cached
            else:
                await self.start_ttfb_metrics()
                pcm = await self._synth(text)
                await self.stop_ttfb_metrics()
        except Exception as exc:
            yield ErrorFrame(f"tts failed: {exc}")
            yield TTSStoppedFrame()
            return

        chunk = self.sample_rate // 10 * 2      # 100 ms of 16-bit mono
        for i in range(0, len(pcm), chunk):
            yield TTSAudioRawFrame(pcm[i:i + chunk], self.sample_rate, 1, context_id=context_id)
        yield TTSStoppedFrame()
