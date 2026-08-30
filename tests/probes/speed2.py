"""Find the speed that sounds natural and still transcribes cleanly."""
import io, os, sys, importlib.util, difflib, time
import numpy as np, soundfile as sf, torch, requests
from omnivoice.models.omnivoice import OmniVoiceGenerationConfig as GC

TXT = "नमस्ते, मैं आपकी कैसे मदद कर सकता हूँ।"
REF = os.path.expanduser("~/hindi_sample.wav")
RT  = "मेजरकेन खाना भूमध्य सागर में समान क्षेत्रों की तरह रोटी सब्जियों और मांस विशेष रूप से होता है"

torch.cuda.is_available = lambda: False
D = os.path.expanduser("~/models/IndicConformer")
sys.path.insert(0, D)
spec = importlib.util.spec_from_file_location("iasr", os.path.join(D, "model_onnx.py"))
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
asr = mod.IndicASRModel(mod.IndicASRConfig(ts_folder=D, FRAME_DURATION_MS=0.04))
torch.cuda.is_available = lambda: True

def heard(a):
    i = np.linspace(0, len(a)-1, int(len(a)*16000/24000))
    b = np.interp(i, np.arange(len(a)), a).astype("float32")
    o = asr(torch.tensor(b).unsqueeze(0), "hi", "rnnt")
    return str(o[0] if isinstance(o,(list,tuple)) else o).strip()

from omnivoice import OmniVoice
m = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cuda:0", dtype=torch.float16)
m.generate(text="नमस्ते।", ref_audio=REF, ref_text=RT, generation_config=GC(num_step=16))

print(f"\n{'speed':>6} {'ms':>6} {'sec':>5} {'match':>6}  ASR")
for sp in (None, 1.3, 1.6, 2.0, 2.4):
    kw = {} if sp is None else {"speed": sp}
    t0 = time.perf_counter()
    o = m.generate(text=TXT, ref_audio=REF, ref_text=RT,
                   generation_config=GC(num_step=16), **kw)
    dt = (time.perf_counter()-t0)*1000
    a = np.asarray(o[0] if isinstance(o,(list,tuple)) else o, dtype=np.float32)
    h = heard(a)
    r = difflib.SequenceMatcher(None, TXT.replace("।",""), h.replace("।","")).ratio()
    print(f"{str(sp):>6} {dt:6.0f} {len(a)/24000:5.2f} {r:6.2f}  {h[:50]}")
    sf.write(os.path.expanduser(f"~/sp_{sp}.wav"), a, 24000)
print("\n(insaan ~2.0-2.5 s me bolta hai)")
