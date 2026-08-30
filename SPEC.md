# VoAIs Indic Calling Agent
Region: asia-south1 (Mumbai) | Zone: asia-south1-c
Project: hidrogen-500914 | Billing: 010D07-FF0401-07246F (iiml-credit, INR)

## HARD CONSTRAINT
Is billing account pe 1x L4 hi max hai. Verified:
  RTX PRO 6000 (G4)  quota request -> DENIED (2 baar)
  L4  1 -> 4         quota request -> DENIED
  H100/H200/B200                   -> sab 0
  GPUS_ALL_REGIONS                 -> approved at 1 (isse zyada nahi)
Credits-based account pe GPU quota nahi badhta. Zyada GPU chahiye to
card-wala paid billing account lagana padega.

## Node A  (bana diya)
  voais-gpu-1 | g2-standard-12 | 1x NVIDIA L4 24GB | asia-south1-c
  image: pytorch-2-9-cu129-ubuntu-2204-nvidia-580 | disk 200GB pd-balanced

  IndicConformer STT  -> 8001   0.1 GB   22 langs, ONNX
  Gemma-4-E4B-it-qat  -> 8000  11.5 GB   brain (audio bhi le sakta hai)
  IndicF5 TTS         -> 8002   1.4 GB   11 langs (gated, token chahiye)
  VRAM: ~13 GB / 24 GB

## Node B  [CHHUA NAHI]
  e2-standard-4, public static IP. Asterisk + orchestrator + VoAIs app.

## Latency — is hardware pe
  ~860 ms   untuned          <- pehla measurement yahi hoga
  ~600 ms   tuning ke baad   <- realistic target
  ~500 ms   is account pe NAHI ho sakta

  Tuning levers (hardware se nahi, software se):
    endpointing 300 -> 150   semantic turn detection
    TTS         200 -> 120   pehle 3-4 shabd pe bolna shuru
    network      90 ->  60   jitter buffer tight

  500ms ke liye chahiye: 2x L4 (TP) ya full-duplex architecture
  Dono blocked hain -> billing account badlo

## Qwen3.6-35B-A3B kyun nahi
  25.5 GB AWQ, 24 GB card. Fit nahi hota. G4 milta to chalta.

## Cost
  budget cap: 8800 INR/month, alerts 50/80/100%   [laga diya]
  g2-standard-12 ~ INR 80/hr -> kaam khatam hote hi VM STOP karna hai

## NOTE
  instance-20260815-062935 (e2-standard-2) 15 Aug se chal raha hai.
  Mera banaya nahi. Zaroorat na ho to band kar dena.
