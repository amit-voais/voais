#!/usr/bin/env bash
# Creates a SPOT L4 VM + a persistent model disk. Spot = ~70% cheaper.
# Run 00_preflight.sh first. Set a budget alert before running this.
set -euo pipefail
source "$(dirname "$0")/../.env"

echo ">> Creating persistent model disk (survives VM deletion)"
gcloud compute disks create "$MODEL_DISK" \
  --project="$PROJECT_ID" --zone="$ZONE" \
  --size=60GB --type=pd-balanced 2>/dev/null || echo "   disk already exists, reusing"

# Preflight the global quota - regional NVIDIA_L4_GPUS being 1 is NOT enough.
GAR=$(gcloud compute project-info describe --project="$PROJECT_ID" \
  --format="value(quotas.filter(metric:GPUS_ALL_REGIONS).extract(limit))" 2>/dev/null | tr -d '[]')
if [ "${GAR:-0}" = "0" ] || [ -z "${GAR:-}" ]; then
  echo "BLOCKED: GPUS_ALL_REGIONS quota is 0. No GPU VM can be created in any region."
  echo "Request an increase to 1 here, then re-run:"
  echo "  https://console.cloud.google.com/iam-admin/quotas?project=$PROJECT_ID&metric=compute.googleapis.com%2Fgpus_all_regions"
  exit 1
fi

echo ">> Creating SPOT g2-standard-4 (1x L4)"
gcloud compute instances create "$VM_NAME" \
  --project="$PROJECT_ID" --zone="$ZONE" \
  --machine-type=g2-standard-4 \
  --provisioning-model=SPOT \
  --instance-termination-action=STOP \
  --maintenance-policy=TERMINATE \
  --image-family=common-cu129-ubuntu-2204-nvidia-580 \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=80GB --boot-disk-type=pd-balanced \
  --disk="name=$MODEL_DISK,device-name=models,mode=rw,boot=no" \
  --metadata="install-nvidia-driver=True" \
  --no-address \
  --scopes=cloud-platform

cat <<'NOTE'

VM created with NO public IP (per SPEC). Reach it over IAP:

  gcloud compute ssh VM_NAME --zone ZONE --tunnel-through-iap

If IAP SSH is blocked, allow it once:

  gcloud compute firewall-rules create allow-iap-ssh \
    --network=default --direction=INGRESS --action=allow \
    --rules=tcp:22 --source-ranges=35.235.240.0/20

To curl the model ports from your laptop, port-forward instead of opening firewall:

  gcloud compute ssh VM_NAME --zone ZONE --tunnel-through-iap -- \
    -L 8000:localhost:8000 -L 8001:localhost:8001 -L 8002:localhost:8002

NOTE
echo "REMEMBER: scripts/99_stop_vm.sh when you finish."
