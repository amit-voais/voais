#!/usr/bin/env bash
# Deletes EVERYTHING this project created: VM, model disk, firewall rule.
# After this runs, billing for the project goes back to zero.
# Pass --yes to skip the confirmation prompt.
set -uo pipefail
source "$(dirname "$0")/../.env"

echo "About to DELETE from project $PROJECT_ID:"
echo "  - instance : $VM_NAME (zone $ZONE)"
echo "  - disk     : $MODEL_DISK (zone $ZONE)  <-- 11.8 GB of models, gone"
echo "  - router   : voais-router + voais-nat (Cloud NAT)"
echo

if [ "${1:-}" != "--yes" ]; then
  read -r -p "Type DELETE to confirm: " ans
  [ "$ans" = "DELETE" ] || { echo "Aborted. Nothing deleted."; exit 1; }
fi

echo ">> deleting instance"
gcloud compute instances delete "$VM_NAME" --zone="$ZONE" --project="$PROJECT_ID" --quiet 2>&1 \
  | sed 's/^/   /' || echo "   (not found, skipping)"

echo ">> deleting model disk"
gcloud compute disks delete "$MODEL_DISK" --zone="$ZONE" --project="$PROJECT_ID" --quiet 2>&1 \
  | sed 's/^/   /' || echo "   (not found, skipping)"

echo ">> deleting Cloud NAT + router"
gcloud compute routers nats delete voais-nat --router=voais-router --region="$REGION" \
  --project="$PROJECT_ID" --quiet 2>&1 | sed 's/^/   /' || echo "   (not found)"
gcloud compute routers delete voais-router --region="$REGION" --project="$PROJECT_ID" --quiet 2>&1 \
  | sed 's/^/   /' || echo "   (not found)"

echo
echo "=== VERIFY: anything left that still bills? ==="
echo "-- instances --"
gcloud compute instances list --project="$PROJECT_ID" --format="table(name,zone,status)" 2>&1 | sed 's/^/   /'
echo "-- disks --"
gcloud compute disks list --project="$PROJECT_ID" --format="table(name,zone,sizeGb)" 2>&1 | sed 's/^/   /'
echo "-- external IPs --"
gcloud compute addresses list --project="$PROJECT_ID" --format="table(name,region,status)" 2>&1 | sed 's/^/   /'
echo
echo "If all three are empty, nothing is billing. Done."
