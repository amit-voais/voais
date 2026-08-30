"""indic-parler-tts: latency and intelligibility, same metric as the others."""
import os, sys, importlib.util, difflib, time
import numpy as np, soundfile as sf, torch

SENTS = ["नमस्ते, मैं आपकी कैसे मदद कर सकता हूँ।",
         "जी बिल्कुल, कल सुबह दस बजे का समय बुक कर देती हूँ।"]
DESC = "A female speaker with a clear voice speaks at a moderate pace in a quiet room."

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

from transformers import AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration
M = "ai4bharat/indic-parler-tts"
print("loading...", flush=True)
t0 = time.perf_counter()
m = ParlerTTSForConditionalGeneration.from_pretrained(M, torch_dtype=torch.float16).to("cuda").eval()
tok = AutoTokenizer.from_pretrained(M)
dtok = AutoTokenizer.from_pretrained(m.config.text_encoder._name_or_path)
sr_ = m.config.sampling_rate
print(f"  loaded {time.perf_counter()-t0:.1f}s, sr={sr_}", flush=True)

# warm
d = dtok(DESC, return_tensors="pt").to("cuda"); p = tok("नमस्ते।", return_tensors="pt").to("cuda")
with torch.no_grad():
    m.generate(input_ids=d.input_ids, attention_mask=d.attention_mask,
               prompt_input_ids=p.input_ids, prompt_attention_mask=p.attention_mask)

print(f"\n{'ms':>7} {'sec':>5} {'match':>6}  ASR")
for t in SENTS:
    d = dtok(DESC, return_tensors="pt").to("cuda")
    p = tok(t, return_tensors="pt").to("cuda")
    t0 = time.perf_counter()
    with torch.no_grad():
        g = m.generate(input_ids=d.input_ids, attention_mask=d.attention_mask,
                       prompt_input_ids=p.input_ids, prompt_attention_mask=p.attention_mask)
    dt = (time.perf_counter()-t0)*1000
    a = g.cpu().numpy().squeeze().astype("float32")
    h = heard(a, sr_)
    r = difflib.SequenceMatcher(None, t.replace("।",""), h.replace("।","")).ratio()
    print(f"{dt:7.0f} {len(a)/sr_:5.2f} {r:6.2f}  {h[:46]}")
    sf.write(os.path.expanduser("~/parler_out.wav"), a, sr_)
