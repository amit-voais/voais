import time, os, numpy as np, torch
from omnivoice import OmniVoice
REF=os.path.expanduser("~/hindi_sample.wav")
RT="मेजरकेन खाना भूमध्य सागर में समान क्षेत्रों की तरह रोटी सब्जियों और मांस विशेष रूप से होता है"
print("loading...", flush=True)
m=OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cuda:0", dtype=torch.float16)
texts=["जी, मैं आपकी क्या मदद कर सकती हूँ?",
       "जी, मैं आपकी क्या मदद कर सकती हूँ?",
       "ठीक है, कल सुबह दस बजे का समय आपके लिए बुक कर देती हूँ।",
       "नमस्ते।"]
for i,t in enumerate(texts):
    t0=time.perf_counter()
    a=m.generate(text=t, ref_audio=REF, ref_text=RT)
    a=a[0] if isinstance(a,(list,tuple)) else a
    dt=(time.perf_counter()-t0)*1000
    dur=len(np.asarray(a))/24000
    print(f"  run{i+1}: {dt:7.0f} ms  -> {dur:.2f}s audio  (RTF {dt/1000/dur:.2f})  | {t[:34]}", flush=True)
# check knobs
import inspect
print("\ngenerate() params:", list(inspect.signature(m.generate).parameters))
