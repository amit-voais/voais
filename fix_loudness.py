"""Normalise TTS output to a usable level.

The server only ever attenuated (`if peak > 1.0`), so a quiet reference clip
produced quiet speech and nothing brought it back up. Measured on the shipped
audio: peak -23 dBFS, RMS -37 dBFS, against a caller mic at -1.9 dBFS. That
20 dB gap is what sounds like whispering.

RMS-normalise to -18 dBFS, then pull the gain back if that would push the peak
past -1 dBFS.
"""
import os
p = os.path.expanduser("~/tts/omni_server.py")
s = open(p).read()

old = """    a = np.asarray(a, dtype=np.float32)
    peak = float(np.abs(a).max()) or 1.0
    if peak > 1.0:
        a = a / peak"""
new = """    a = np.asarray(a, dtype=np.float32)
    peak = float(np.abs(a).max())
    rms = float(np.sqrt((a ** 2).mean())) if a.size else 0.0
    if rms > 1e-6:
        gain = TARGET_RMS / rms
        if peak * gain > PEAK_CEIL:          # keep headroom, no clipping
            gain = PEAK_CEIL / peak
        a = a * gain
    elif peak > 1.0:
        a = a / peak"""
assert old in s, "normalise anchor missing"
s = s.replace(old, new)

s = s.replace('DEFAULT_STEPS = int(os.environ.get("OMNI_STEPS", "4"))',
              'DEFAULT_STEPS = int(os.environ.get("OMNI_STEPS", "16"))\n'
              'TARGET_RMS = 10 ** (-18 / 20)   # -18 dBFS, normal speech level\n'
              'PEAK_CEIL = 10 ** (-1 / 20)     # -1 dBFS ceiling')

# point at the loudness-corrected reference
s = s.replace('"OMNI_REF", "~/refsil_0.2_6.1.wav"', '"OMNI_REF", "~/ref_loud.wav"')
open(p, "w").write(s)
print("patched omni_server.py")
for k in ("TARGET_RMS", "PEAK_CEIL", "ref_loud"):
    print(f"  {k:12s} x{s.count(k)}")
