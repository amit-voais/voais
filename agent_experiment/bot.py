"""VoAIs voice agent on Pipecat.

Replaces the hand-rolled orchestration (browser RMS VAD, HTTP + SSE turns,
sequential audio chunks) with Pipecat's pipeline. What that buys, in order of
how much it matters on a real call:

  * barge-in - the caller can interrupt mid-sentence, which the previous build
    could not do at all and which people notice more than latency
  * Silero VAD and proper turn handling instead of an RMS threshold
  * a `livekit` transport, so SIP telephony is a transport swap rather than a
    rewrite - the models below do not change

The models stay self-hosted: audio never leaves the VM, which is the whole
argument for this stack over a hosted realtime API.

Run locally against a GPU box:
    VLLM_URL=http://localhost:8000 TTS_URL=http://localhost:8002 python bot.py
(port-forward both through the IAP tunnel first)
"""
import os

from dotenv import load_dotenv
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.turns.user_start.vad_user_turn_start_strategy import VADUserTurnStartStrategy
from pipecat.turns.user_stop.speech_timeout_user_turn_stop_strategy import (
    SpeechTimeoutUserTurnStopStrategy,
)
from pipecat.turns.user_turn_processor import UserTurnProcessor
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import TransportParams

from voais.stt import GemmaSTTService
from voais.tts import IndicF5TTSService

load_dotenv()

VLLM_URL = os.getenv("VLLM_URL", "http://localhost:8000")
TTS_URL = os.getenv("TTS_URL", "http://localhost:8002")
NFE = int(os.getenv("F5_NFE_STEP", "8"))
# Silence after the caller stops. Pure dead time on every turn, so it is tuned
# down from the 0.8 s default; below ~0.3 s the agent starts cutting people off.
STOP_SECS = float(os.getenv("VAD_STOP_SECS", "0.4"))

# The single most important line in this file. IndicF5 is trained on Devanagari;
# given romanised Hindi it emits confident gibberish at about half the correct
# duration. The model sounds broken when the prompt is what is actually broken.
SYSTEM = """Aap VoAIs ke call center agent hain.

SABSE ZAROORI NIYAM: Jawab HAMESHA Devanagari lipi mein likhiye.
Sahi:  नमस्ते, मैं आपकी क्या मदद कर सकता हूँ?
Galat: Namaste, main aapki kya madad kar sakta hoon?
Roman ya English akshar bilkul mat likhiye, chahe user Roman mein likhe.
Angrezi shabd bhi Devanagari mein: email -> ईमेल, account -> अकाउंट.

Phone par baat kar rahe hain: ek ya do chhote vakya. Koi markdown, koi list,
koi emoji nahi."""

GREETING = "नमस्ते, वॉयस ए आई में आपका स्वागत है। मैं आपकी क्या मदद कर सकता हूँ?"

# Synthesised once at startup and replayed instantly. The fixed per-utterance
# cost of a flow-matching TTS cannot be optimised away, so the opening line is
# simply paid for in advance.
CACHED = (GREETING, "जी बिल्कुल।", "एक मिनट।", "जी, समझ गया।", "माफ़ कीजिए, दोबारा बोलिए।")


def build_pipeline(transport):
    stt = GemmaSTTService(base_url=VLLM_URL)
    tts = IndicF5TTSService(base_url=TTS_URL, nfe=NFE, cache_phrases=CACHED)
    llm = OpenAILLMService(
        api_key="not-needed",          # vLLM ignores it but the client demands one
        base_url=f"{VLLM_URL}/v1",
        settings=OpenAILLMService.Settings(model="voais-llm"),
    )

    context = LLMContext([{"role": "system", "content": SYSTEM}])
    agg = LLMContextAggregatorPair(context)

    # In Pipecat 1.8 turn handling lives in the pipeline, not on the transport -
    # passing vad_analyzer to TransportParams is silently ignored by pydantic and
    # leaves the agent with no turn detection at all.
    vad = VADProcessor(
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=STOP_SECS)),
    )
    turns = UserTurnProcessor(
        user_turn_strategies=UserTurnStrategies(
            # enable_interruptions is barge-in: the caller can cut the agent off
            # mid-sentence, which the previous hand-rolled build could not do.
            start=[VADUserTurnStartStrategy(enable_interruptions=True)],
            stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=STOP_SECS)],
        ),
    )

    pipeline = Pipeline([
        transport.input(),
        vad,
        stt,
        turns,
        agg.user(),
        llm,
        tts,
        transport.output(),
        agg.assistant(),
    ])

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    @transport.event_handler("on_client_connected")
    async def _connected(_t, _c):
        logger.info("caller connected - greeting from cache")
        await task.queue_frames([agg.user().get_context_frame()])

    @transport.event_handler("on_client_disconnected")
    async def _disconnected(_t, _c):
        logger.info("caller disconnected")
        await task.cancel()

    return task


def transport_params() -> TransportParams:
    return TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        audio_out_sample_rate=24000,   # IndicF5 native rate; avoids a resample
    )


async def run(transport):
    task = build_pipeline(transport)
    await PipelineRunner(handle_sigint=False).run(task)


if __name__ == "__main__":
    from voais.server import serve
    logger.info(f"vLLM={VLLM_URL}  TTS={TTS_URL}  nfe={NFE}")
    serve(run, transport_params)
