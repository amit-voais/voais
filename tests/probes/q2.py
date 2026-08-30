"""Read the TTS back with the ASR - no listening needed, no extra GPU."""
import io, os, sys, importlib.util, difflib, time
import numpy as np, soundfile as sf, torch, requests

TARGET = "नमस्ते, मैं आपकी कैसे मदद कर सकता हूँ।"
torch.cuda.is_available = lambda: False          # ASR on CPU; GPU is full
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

print(f"\ntarget: {TARGET}\n")
print(f"{'steps':>6} {'ms':>6} {'sec':>5} {'match':>6}  ASR ne kya suna")
for ns in (2, 4, 8, 16, 32):
    t0 = time.perf_counter()
    r = requests.post("http://localhost:8002/synthesize",
                      json={"text": TARGET, "language": "hi", "nfe": ns}, timeout=180)
    dt = (time.perf_counter()-t0)*1000
    a, sr_ = sf.read(io.BytesIO(r.content), dtype="float32")
    if a.ndim > 1: a = a.mean(axis=1)
    h = heard(a, sr_)
    m = difflib.SequenceMatcher(None, TARGET.replace("।",""), h.replace("।","")).ratio()
    print(f"{ns:>6} {dt:6.0f} {len(a)/sr_:5.2f} {m:6.2f}  {h[:60]}")
    sf.write(os.path.expanduser(f"~/q_{ns}.wav"), a, sr_)
