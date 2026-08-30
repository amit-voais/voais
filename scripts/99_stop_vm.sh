#!/usr/bin/env bash
# Stop the VM. Run this after EVERY session. A running L4 bills ~Rs 20/day idle.
set -euo pipefail
source "$(dirname "$0")/../.env"
gcloud compute instances stop "$VM_NAME" --zone="$ZONE" --project="$PROJECT_ID"
echo "Stopped $VM_NAME. Disk still billed (~\$0.04/GB/month); models preserved."
