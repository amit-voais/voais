"""Objective TTS quality: synthesize, then read it back with the ASR.

If the recogniser cannot recover the sentence, the audio is not intelligible -
no listening required.
"""
import io, os, sys, importlib.util, difflib, time
import numpy as np, soundfile as sf, torch, requests

TARGET = "नमस्ते, मैं आपकी कैसे मदद कर सकता हूँ।"
REF_FULL = os.path.expanduser("~/hindi_sample.wav")
REF_TEXT = "मेजरकेन खाना भूमध्य सागर में समान क्षेत्रों की तरह रोटी सब्जियों और मांस विशेष रूप से होता है"

# a shorter reference clip - long refs degrade zero-shot cloning
d, sr = sf.read(REF_FULL, dtype="float32")
if d.ndim > 1: d = d.mean(axis=1)
short = d[: int(sr * 5)]
REF_SHORT = os.path.expanduser("~/ref_short.wav")
sf.write(REF_SHORT, short, sr)
REF_TEXT_SHORT = "मेजरकेन खाना भूमध्य सागर में समान क्षेत्रों की तरह"

# the streaming server already holds the GPU copy; run this check on CPU
_real = torch.cuda.is_available
torch.cuda.is_available = lambda: False
D = os.path.expanduser("~/models/IndicConformer")
sys.path.insert(0, D)
spec = importlib.util.spec_from_file_location("iasr", os.path.join(D, "model_onnx.py"))
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
asr = mod.IndicASRModel(mod.IndicASRConfig(ts_folder=D, FRAME_DURATION_MS=0.04))

torch.cuda.is_available = _real
from omnivoice import OmniVoice
from omnivoice.models.omnivoice import OmniVoiceGenerationConfig as GC
m = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cuda:0", dtype=torch.float16)
m.generate(text="नमस्ते।", ref_audio=REF_SHORT, ref_text=REF_TEXT_SHORT, generation_config=GC(num_step=8))

def score(a, sr_out):
    if sr_out != 16000:
        i = np.linspace(0, len(a)-1, int(len(a)*16000/sr_out))
        a = np.interp(i, np.arange(len(a)), a).astype("float32")
    out = asr(torch.tensor(a).unsqueeze(0), "hi", "rnnt")
    txt = str(out[0] if isinstance(out,(list,tuple)) else out).strip()
    return difflib.SequenceMatcher(None, TARGET.replace("।",""), txt.replace("।","")).ratio(), txt

print(f"\ntarget: {TARGET}\n")
print(f"{'ref':>6} {'steps':>6} {'ms':>6} {'match':>6}  heard")
for rname, rwav, rtxt in (("12s", REF_FULL, REF_TEXT), ("5s", REF_SHORT, REF_TEXT_SHORT)):
    for ns in (4, 8, 16, 32):
        t0 = time.perf_counter()
        o = m.generate(text=TARGET, ref_audio=rwav, ref_text=rtxt,
                       generation_config=GC(num_step=ns))
        dt = (time.perf_counter()-t0)*1000
        a = np.asarray(o[0] if isinstance(o,(list,tuple)) else o, dtype=np.float32)
        r, heard = score(a, 24000)
        print(f"{rname:>6} {ns:>6} {dt:6.0f} {r:6.2f}  {heard[:64]}")
        sf.write(os.path.expanduser(f"~/q_{rname}_{ns}.wav"), a, 24000)
