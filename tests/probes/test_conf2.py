import os, sys, time, importlib.util, inspect
import numpy as np, soundfile as sf, torch

D = os.path.expanduser("~/models/IndicConformer")
sys.path.insert(0, D)
spec = importlib.util.spec_from_file_location("iasr", os.path.join(D, "model_onnx.py"))
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

print("constructing directly (their from_pretrained only accepts a hub id)...", flush=True)
t0 = time.perf_counter()
cfg = mod.IndicASRConfig(ts_folder=D, FRAME_DURATION_MS=0.04)
m = mod.IndicASRModel(cfg)
print(f"  loaded in {time.perf_counter()-t0:.1f}s", flush=True)

d, sr = sf.read(os.path.expanduser("~/hindi_sample.wav"), dtype="float32")
if d.ndim > 1: d = d.mean(axis=1)
if sr != 16000:
    idx = np.linspace(0, len(d)-1, int(len(d)*16000/sr))
    d = np.interp(idx, np.arange(len(d)), d).astype("float32")
print(f"  audio {len(d)/16000:.2f}s")
try: print("  forward signature:", inspect.signature(m.forward))
except Exception: pass

x = torch.tensor(d).unsqueeze(0)
for mode in ("ctc", "rnnt"):
    try:
        t0 = time.perf_counter(); out = m(x, "hi", mode); dt = (time.perf_counter()-t0)*1000
        txt = out[0] if isinstance(out, (list, tuple)) else out
        print(f"\n[{mode}] {dt:7.0f} ms  RTF {dt/1000/(len(d)/16000):.3f}")
        print(f"   -> {str(txt)[:150]}")
    except Exception as e:
        print(f"\n[{mode}] FAILED {type(e).__name__}: {str(e)[:200]}")
