"""smart-turn-v2 with its real head.

AutoModelForAudioClassification silently replaces this checkpoint's head with a
randomly initialised one - it loads, it runs, and every probability it returns
is noise. The layer shapes below are read straight off the checkpoint, and the
load is verified tensor by tensor rather than trusted.
"""
import os, torch, torch.nn as nn
from safetensors.torch import load_file
from transformers import Wav2Vec2Model, Wav2Vec2Config, AutoFeatureExtractor

DIR = os.path.expanduser("~/models/smart-turn-v2")


class TurnDetector(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.wav2vec2 = Wav2Vec2Model(cfg)
        h = cfg.hidden_size                       # 768
        self.pool_attention = nn.Sequential(
            nn.Linear(h, 256), nn.Tanh(), nn.Linear(256, 1))
        self.classifier = nn.Sequential(
            nn.Linear(h, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, input_values, attention_mask=None):
        hs = self.wav2vec2(input_values, attention_mask=attention_mask).last_hidden_state
        w = torch.softmax(self.pool_attention(hs), dim=1)     # (B,T,1)
        pooled = (hs * w).sum(dim=1)                          # (B,H)
        return self.classifier(pooled)                        # (B,1) logit


def load(device="cuda"):
    cfg = Wav2Vec2Config.from_pretrained(DIR)
    m = TurnDetector(cfg)
    sd = load_file(os.path.join(DIR, "model.safetensors"))
    missing, unexpected = m.load_state_dict(sd, strict=False)
    # a silently random head is the failure mode here, so prove it did not happen
    head = [k for k in sd if k.startswith(("classifier.", "pool_attention."))]
    bad = [k for k in head if k in missing]
    if bad or len(head) != 12:
        raise RuntimeError(f"head did not load: missing={bad} found={len(head)}")
    live = dict(m.named_parameters())
    good = sum(1 for k in head if torch.allclose(sd[k].float(), live[k].detach().float(), atol=1e-6))
    if good != len(head):
        raise RuntimeError(f"head verification failed: {good}/{len(head)}")
    print(f"  head verified: {good}/{len(head)} tensors, missing={len(missing)} unexpected={len(unexpected)}")
    fe = AutoFeatureExtractor.from_pretrained(DIR)
    return m.to(device).eval(), fe
