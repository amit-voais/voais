"""Cut the reference at silence on both ends.

OmniVoice continues from the reference, so a clip that stops mid-phrase leaks
its last words into the start of every generation - the caller hears
"...karta hai" before the actual reply.
"""
import os, sys, importlib.util, difflib, time
import numpy as np, soundfile as sf, torch

TXT = "नमस्ते, मैं आपकी कैसे मदद कर सकता हूँ।"
FIRST = "नमस्ते"
FULL = os.path.expanduser("~/hindi_sample.wav")

torch.cuda.is_available = lambda: False
D = os.path.expanduser("~/models/IndicConformer")
sys.path.insert(0, D)
spec = importlib.util.spec_from_file_location("iasr", os.path.join(D, "model_onnx.py"))
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
asr = mod.IndicASRModel(mod.IndicASRConfig(ts_folder=D, FRAME_DURATION_MS=0.04))
torch.cuda.is_available = lambda: True

def heard(a, sr_in=24000):
    i = np.linspace(0, len(a)-1, int(len(a)*16000/sr_in))
    b = np.interp(i, np.arange(len(a)), a).astype("float32")
    o = asr(torch.tensor(b).unsqueeze(0), "hi", "rnnt")
    return str(o[0] if isinstance(o,(list,tuple)) else o).strip()

d, sr = sf.read(FULL, dtype="float32")
if d.ndim > 1: d = d.mean(axis=1)

# energy profile -> silence gaps long enough to cut on
win = int(0.02*sr)
e = np.array([np.sqrt((d[i:i+win]**2).mean()) for i in range(0, len(d)-win, win)])
thr = max(e.max()*0.06, 0.006)
sil = e < thr
gaps = []
i = 0
while i < len(sil):
    if sil[i]:
        j = i
        while j < len(sil) and sil[j]: j += 1
        if (j-i)*0.02 >= 0.18: gaps.append(((i+j)//2*0.02, (j-i)*0.02))
        i = j
    else: i += 1
print("silence gaps (sec, len):", [(round(a,2), round(b,2)) for a,b in gaps][:10])

cands = []
for a0, _ in gaps:
    for a1, _ in gaps:
        if 3.0 <= a1-a0 <= 7.0: cands.append((a0, a1))
cands = cands[:4]
print("candidates:", [(round(a,2), round(b,2)) for a,b in cands])

from omnivoice import OmniVoice
from omnivoice.models.omnivoice import OmniVoiceGenerationConfig as GC
m = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cuda:0", dtype=torch.float16)

print(f"\n{'cut':>12} {'out_s':>6} {'match':>6} {'prefix?':>8}  ASR")
best = None
for a0, a1 in cands:
    seg = d[int(a0*sr):int(a1*sr)]
    p = os.path.expanduser(f"~/refsil_{a0:.1f}_{a1:.1f}.wav"); sf.write(p, seg, sr)
    rt = heard(seg, sr)
    if not rt.strip(): continue
    o = m.generate(text=TXT, ref_audio=p, ref_text=rt, generation_config=GC(num_step=16))
    a = np.asarray(o[0] if isinstance(o,(list,tuple)) else o, dtype=np.float32)
    h = heard(a)
    r = difflib.SequenceMatcher(None, TXT.replace("।",""), h.replace("।","")).ratio()
    pre = "NO" if h.startswith(FIRST) else "YES:" + h.split()[0][:8]
    print(f"{a0:5.1f}-{a1:5.1f} {len(a)/24000:6.2f} {r:6.2f} {pre:>8}  {h[:42]}")
    if h.startswith(FIRST) and (best is None or r > best[0]):
        best = (r, p, rt, len(a)/24000)
if best:
    print(f"\nBEST -> {best[1]}  match={best[0]:.2f}  {best[3]:.2f}s")
    print(f"REF_TEXT: {best[2]}")
    open(os.path.expanduser("~/best_ref.txt"),"w").write(f"{best[1]}\n{best[2]}\n")
else:
    print("\nkoi bhi prefix-free nahi mila")
