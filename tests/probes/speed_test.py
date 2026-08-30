"""Is it the sample rate or the speaking rate? Measure both."""
import io, os, sys, importlib.util, difflib, time
import numpy as np, soundfile as sf, torch, requests

TXT = "नमस्ते, मैं आपकी कैसे मदद कर सकता हूँ।"
# a human reads this in roughly 2.0-2.5 s
torch.cuda.is_available = lambda: False
D = os.path.expanduser("~/models/IndicConformer")
sys.path.insert(0, D)
spec = importlib.util.spec_from_file_location("iasr", os.path.join(D, "model_onnx.py"))
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
asr = mod.IndicASRModel(mod.IndicASRConfig(ts_folder=D, FRAME_DURATION_MS=0.04))

def heard(a, sr_in):
    if sr_in != 16000:
        i = np.linspace(0, len(a)-1, int(len(a)*16000/sr_in))
        a = np.interp(i, np.arange(len(a)), a).astype("float32")
    o = asr(torch.tensor(a).unsqueeze(0), "hi", "rnnt")
    return str(o[0] if isinstance(o,(list,tuple)) else o).strip()

r = requests.post("http://localhost:8002/synthesize",
                  json={"text": TXT, "language": "hi", "nfe": 16}, timeout=180)
a, sr_ = sf.read(io.BytesIO(r.content), dtype="float32")
if a.ndim > 1: a = a.mean(axis=1)
print(f"server WAV : {sr_} Hz, {len(a)} samples, {len(a)/sr_:.2f} s")
print(f"  agar asli rate 48k hota to duration hoti: {len(a)/48000:.2f} s")
print(f"  insaan is vaakya ko bolta hai: ~2.0-2.5 s")
h = heard(a, sr_)
print(f"  ASR (24k maan kar): {difflib.SequenceMatcher(None, TXT.replace('।',''), h.replace('।','')).ratio():.2f}  {h[:55]}")
h2 = heard(a, 48000)
print(f"  ASR (48k maan kar): {difflib.SequenceMatcher(None, TXT.replace('।',''), h2.replace('।','')).ratio():.2f}  {h2[:55]}")
sf.write(os.path.expanduser("~/spd_base.wav"), a, sr_)
