"""The clone copies the reference's pace. Try slices and pick the fastest
one that still transcribes cleanly."""
import os, sys, importlib.util, difflib, time
import numpy as np, soundfile as sf, torch

TXT = "नमस्ते, मैं आपकी कैसे मदद कर सकता हूँ।"
FULL = os.path.expanduser("~/hindi_sample.wav")
RT_FULL = "मेजरकेन खाना भूमध्य सागर में समान क्षेत्रों की तरह रोटी सब्जियों और मांस विशेष रूप से होता है"

torch.cuda.is_available = lambda: False
D = os.path.expanduser("~/models/IndicConformer")
sys.path.insert(0, D)
spec = importlib.util.spec_from_file_location("iasr", os.path.join(D, "model_onnx.py"))
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
asr = mod.IndicASRModel(mod.IndicASRConfig(ts_folder=D, FRAME_DURATION_MS=0.04))
torch.cuda.is_available = lambda: True

def heard16(a, sr_in):
    i = np.linspace(0, len(a)-1, int(len(a)*16000/sr_in))
    b = np.interp(i, np.arange(len(a)), a).astype("float32")
    o = asr(torch.tensor(b).unsqueeze(0), "hi", "rnnt")
    return str(o[0] if isinstance(o,(list,tuple)) else o).strip()

d, sr = sf.read(FULL, dtype="float32")
if d.ndim > 1: d = d.mean(axis=1)

# slice the reference, and transcribe each slice so ref_text actually matches it
slices = {"0-5": (0,5), "3-8": (3,8), "6-11": (6,11), "full": (0, len(d)/sr)}
refs = {}
for name,(a0,a1) in slices.items():
    seg = d[int(a0*sr):int(a1*sr)]
    p = os.path.expanduser(f"~/ref_{name}.wav"); sf.write(p, seg, sr)
    rt = RT_FULL if name == "full" else heard16(seg, sr)
    refs[name] = (p, rt, len(seg)/sr)
    print(f"  ref {name:5s} {len(seg)/sr:5.2f}s  text: {rt[:50]}")

from omnivoice import OmniVoice
from omnivoice.models.omnivoice import OmniVoiceGenerationConfig as GC
m = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cuda:0", dtype=torch.float16)
m.generate(text="नमस्ते।", ref_audio=refs["0-5"][0], ref_text=refs["0-5"][1],
           generation_config=GC(num_step=16))

print(f"\n{'ref':>6} {'ms':>6} {'out_s':>6} {'match':>6}  ASR")
for name,(p, rt, rdur) in refs.items():
    if not rt.strip(): continue
    t0 = time.perf_counter()
    o = m.generate(text=TXT, ref_audio=p, ref_text=rt, generation_config=GC(num_step=16))
    dt = (time.perf_counter()-t0)*1000
    a = np.asarray(o[0] if isinstance(o,(list,tuple)) else o, dtype=np.float32)
    h = heard16(a, 24000)
    r = difflib.SequenceMatcher(None, TXT.replace("।",""), h.replace("।","")).ratio()
    print(f"{name:>6} {dt:6.0f} {len(a)/24000:6.2f} {r:6.2f}  {h[:48]}")
    sf.write(os.path.expanduser(f"~/out_{name}.wav"), a, 24000)
print("\n(target ~2.0-2.5 s)")
