# Voice reference

The TTS clones its voice, pace and loudness from a single reference clip. The
clip itself is not in this repo — wav files are gitignored because they carry
real voices.

Supply your own: 5–10 s of Hindi, spoken conversationally rather than read, cut
at silence on both ends, RMS around -18 dBFS.

All four voice defects we hit traced back to a bad reference — a research-dataset
clip that was quiet (-22 dBFS), slowly read, and cut mid-phrase. See FINDINGS.md.
