"""Generate the same sentences from IndicF5 and indic-parler-tts, then listen.

Voice quality is not settled by benchmarks. This writes paired WAVs so you can
A-B them yourself in the languages you actually sell in.

Usage:
    python tests/tts_listen_test.py --out data/tts_compare
"""
import argparse
import os

import requests

SENTENCES = [
    ("hi", "Namaste, main VoAIs se bol raha hoon. Aapka appointment kal subah gyarah baje ka hai."),
    ("ta", "Vanakkam, ungal sandhippu naalai kaalai pathinoru manikku."),
    ("te", "Namaskaram, mee appointment repu udayam padakonda gantalaku undi."),
    ("bn", "Nomoskar, apnar appointment kal sokal egaroy tar somoy."),
    ("mr", "Namaskar, tumchi bhet udya sakali akra vajta aahe."),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/tts_compare")
    ap.add_argument("--url", default="http://localhost:8002/synthesize")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    for lang, text in SENTENCES:
        try:
            r = requests.post(args.url, json={"text": text, "language": lang}, timeout=120)
            r.raise_for_status()
            path = os.path.join(args.out, f"{lang}.wav")
            with open(path, "wb") as fh:
                fh.write(r.content)
            print(f"  wrote {path}")
        except Exception as exc:
            print(f"  {lang} FAILED: {exc}")

    print(f"\nListen to everything in {args.out} and judge for yourself.")
    print("Score each on: naturalness, pronunciation, and whether it sounds")
    print("like the same speaker every time. Consistency matters most on calls.")


if __name__ == "__main__":
    main()
