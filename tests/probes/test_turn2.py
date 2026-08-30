import os, sys, time, numpy as np, soundfile as sf, torch, requests
sys.path.insert(0, os.path.expanduser("~"))
from turn_model import load

MAXS = 8  # model window, seconds
print("loading turn detector...", flush=True)
m, fe = load()

# make complete / incomplete clips with the TTS we already run
CASES = [
    ("PURA    ", "मुझे कल सुबह दस बजे अपॉइंटमेंट चाहिए।"),
    ("PURA    ", "जी हाँ, मेरा नाम अमित है।"),
    ("ADHOORA ", "मुझे कल सुबह"),
    ("ADHOORA ", "जी हाँ, मेरा नाम"),
]
print(f"\n{'case':10s} {'P(khatam)':>10s} {'ms':>6s}  text")
for tag, text in CASES:
    r = requests.post("http://localhost:8002/synthesize",
                      json={"text": text, "language": "hi", "nfe": 8}, timeout=120)
    a, sr = sf.read(__import__("io").BytesIO(r.content), dtype="float32")
    if a.ndim > 1: a = a.mean(axis=1)
    if sr != 16000:
        i = np.linspace(0, len(a)-1, int(len(a)*16000/sr))
        a = np.interp(i, np.arange(len(a)), a).astype("float32")
    a = a[-16000*MAXS:]
    x = fe(a, sampling_rate=16000, return_tensors="pt",
           padding="max_length", max_length=16000*MAXS, truncation=True)
    x = {k: v.to("cuda") for k, v in x.items()}
    t0 = time.perf_counter()
    with torch.no_grad(): lg = m(**x)
    dt = (time.perf_counter()-t0)*1000
    p = torch.sigmoid(lg).item()
    print(f"{tag:10s} {p:10.3f} {dt:6.0f}  {text}")
