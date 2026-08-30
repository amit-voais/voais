"""IndicF5 vs OmniVoice, judged by whether the ASR can read the result back.

Same sentence, same recogniser, same metric. Anything below ~0.9 is audio a
caller would struggle with.
"""
import io, os, sys, importlib.util, difflib, time
import numpy as np, soundfile as sf, torch

SENTS = [
    "नमस्ते, मैं आपकी कैसे मदद कर सकता हूँ।",
    "जी बिल्कुल, कल सुबह दस बजे का समय बुक कर देती हूँ।",
]
REF = os.path.expanduser("~/refsil_0.2_6.1.wav")
RT  = "मेजर किन खाना भूमध्य सागर में सामान्य क्षेत्रों की तरह रोटी सब्जियों और मांस"

torch.cuda.is_available = lambda: False
D = os.path.expanduser("~/models/IndicConformer")
sys.path.insert(0, D)
sp = importlib.util.spec_from_file_location("iasr", os.path.join(D, "model_onnx.py"))
mod = importlib.util.module_from_spec(sp); sp.loader.exec_module(mod)
asr = mod.IndicASRModel(mod.IndicASRConfig(ts_folder=D, FRAME_DURATION_MS=0.04))
torch.cuda.is_available = lambda: True

def heard(a, sr_in):
    i = np.linspace(0, len(a)-1, int(len(a)*16000/sr_in))
    b = np.interp(i, np.arange(len(a)), a).astype("float32")
    o = asr(torch.tensor(b).unsqueeze(0), "hi", "rnnt")
    return str(o[0] if isinstance(o,(list,tuple)) else o).strip()

def rate(txt, a, sr_):
    h = heard(a, sr_)
    m = difflib.SequenceMatcher(None, txt.replace("।",""), h.replace("।","")).ratio()
    first_ok = h.startswith(txt.split(",")[0].split()[0])
    return m, len(a)/sr_, h, first_ok

rows = []

# ---------- OmniVoice ----------
from omnivoice import OmniVoice
from omnivoice.models.omnivoice import OmniVoiceGenerationConfig as GC
ov = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cuda:0", dtype=torch.float16)
ov.generate(text="नमस्ते।", ref_audio=REF, ref_text=RT, generation_config=GC(num_step=16))
for ns in (16, 24):
    for t in SENTS:
        t0=time.perf_counter()
        o = ov.generate(text=t, ref_audio=REF, ref_text=RT, generation_config=GC(num_step=ns))
        dt=(time.perf_counter()-t0)*1000
        a = np.asarray(o[0] if isinstance(o,(list,tuple)) else o, np.float32)
        m,d_,h,f = rate(t,a,24000)
        rows.append((f"OmniVoice s={ns}", dt, d_, m, f, h))
del ov; torch.cuda.empty_cache()

# ---------- IndicF5 ----------
try:
    sys.path.insert(0, os.path.expanduser("~/pipecat"))
    os.environ["INDICF5_DIR"] = os.path.expanduser("~/models/IndicF5")
    spec2 = importlib.util.spec_from_file_location("tf", os.path.expanduser("~/pipecat/tts_fast.py"))
    tf = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(tf)
    good = tf._boot()
    print(f"IndicF5 booted, {good} tensors verified", flush=True)
    for nfe in (4, 8, 16):
        for t in SENTS:
            t0=time.perf_counter(); a = tf._synth(t, nfe); dt=(time.perf_counter()-t0)*1000
            m,d_,h,f = rate(t, np.asarray(a,np.float32), 24000)
            rows.append((f"IndicF5 nfe={nfe}", dt, d_, m, f, h))
except Exception as e:
    print("IndicF5 FAILED:", type(e).__name__, str(e)[:200])

print(f"\n{'engine':>16} {'ms':>6} {'sec':>5} {'match':>6} {'clean':>6}  ASR")
for n,dt,d_,m,f,h in rows:
    print(f"{n:>16} {dt:6.0f} {d_:5.2f} {m:6.2f} {('OK' if f else 'PREFIX'):>6}  {h[:44]}")
