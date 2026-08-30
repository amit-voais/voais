"""indic-parler-tts in streaming mode: how soon does the FIRST chunk land?

Blocking generate() took 5132 ms, but that is time-to-complete. Parler is
autoregressive, so the interesting number is time-to-first-chunk.
"""
import os, time, threading
import numpy as np, soundfile as sf, torch
from transformers import AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration, ParlerTTSStreamer

M = "ai4bharat/indic-parler-tts"
TXT = "नमस्ते, मैं आपकी कैसे मदद कर सकता हूँ।"
DESC = "A female speaker with a clear voice speaks at a moderate pace in a quiet room."

print("loading...", flush=True)
t0 = time.perf_counter()
m = ParlerTTSForConditionalGeneration.from_pretrained(M, torch_dtype=torch.float16).to("cuda").eval()
tok = AutoTokenizer.from_pretrained(M)
dtok = AutoTokenizer.from_pretrained(m.config.text_encoder._name_or_path)
sr = m.config.sampling_rate
print(f"  loaded {time.perf_counter()-t0:.1f}s, sr={sr}", flush=True)

frame_rate = m.audio_encoder.config.frame_rate
for play_steps_s in (0.5, 1.0):
    play_steps = int(frame_rate * play_steps_s)
    streamer = ParlerTTSStreamer(m, device="cuda", play_steps=play_steps)
    d = dtok(DESC, return_tensors="pt").to("cuda")
    p = tok(TXT, return_tensors="pt").to("cuda")
    kw = dict(input_ids=d.input_ids, attention_mask=d.attention_mask,
              prompt_input_ids=p.input_ids, prompt_attention_mask=p.attention_mask,
              streamer=streamer, do_sample=True, temperature=1.0)
    t0 = time.perf_counter()
    th = threading.Thread(target=m.generate, kwargs=kw); th.start()
    first, total, n = None, 0, 0
    for chunk in streamer:
        if first is None:
            first = (time.perf_counter()-t0)*1000
        total = (time.perf_counter()-t0)*1000
        n += len(chunk); 
    th.join()
    print(f"\nplay_steps={play_steps_s}s  ->  FIRST CHUNK {first:.0f} ms | complete {total:.0f} ms | {n/sr:.2f}s audio")
