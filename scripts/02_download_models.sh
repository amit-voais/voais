#!/usr/bin/env bash
# RUNS ON THE VM, not your laptop.
# Downloads ~11.8 GB straight from HF onto the mounted model disk.
set -euo pipefail

: "${HF_TOKEN:?Set HF_TOKEN. The ai4bharat models are gated - accept terms first.}"

MNT=/mnt/models
if ! mountpoint -q "$MNT"; then
  echo ">> First run: formatting and mounting the model disk"
  sudo mkdir -p "$MNT"
  if ! sudo blkid /dev/disk/by-id/google-models >/dev/null 2>&1; then
    sudo mkfs.ext4 -F /dev/disk/by-id/google-models
  fi
  sudo mount -o discard,defaults /dev/disk/by-id/google-models "$MNT"
  sudo chown -R "$USER":"$USER" "$MNT"
  echo "/dev/disk/by-id/google-models $MNT ext4 discard,defaults,nofail 0 2" \
    | sudo tee -a /etc/fstab >/dev/null
fi

pip install -q --upgrade "huggingface_hub[cli]"
export HF_HOME="$MNT/hf"

dl () {
  echo ">> $1"
  hf download "$1" --local-dir "$MNT/$(basename "$1")" --token "$HF_TOKEN"
}

dl google/gemma-4-12B-it-qat-w4a16-ct          # 10.3 GB, open
dl ai4bharat/IndicF5                            # 1.4 GB, GATED
dl ai4bharat/indic-conformer-600m-multilingual  # 0.09 GB, GATED

echo
echo "Done. Contents of $MNT:"
du -sh "$MNT"/* 2>/dev/null
