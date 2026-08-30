#!/usr/bin/env bash
# Read-only. Creates nothing, enables nothing, spends nothing.
set -uo pipefail

echo "=== gcloud ==="
if ! command -v gcloud >/dev/null 2>&1; then
  echo "  MISSING. Install: brew install --cask google-cloud-sdk"
  exit 1
fi
gcloud version 2>/dev/null | head -1

echo
echo "=== auth ==="
gcloud auth list --filter=status:ACTIVE --format="value(account)" 2>/dev/null \
  || echo "  not authenticated -> run: gcloud auth login"

echo
echo "=== active project ==="
PROJ=$(gcloud config get-value project 2>/dev/null)
echo "  ${PROJ:-<unset>}"
[ -z "$PROJ" ] && { echo "  set it: gcloud config set project PROJECT_ID"; exit 1; }

echo
echo "=== billing ==="
gcloud beta billing projects describe "$PROJ" \
  --format="value(billingEnabled)" 2>/dev/null || echo "  could not read (needs billing.viewer)"

echo
echo "=== required APIs ==="
for api in compute.googleapis.com iap.googleapis.com; do
  if gcloud services list --enabled --filter="config.name=$api" \
       --format="value(config.name)" 2>/dev/null | grep -q .; then
    echo "  $api  ENABLED"
  else
    echo "  $api  NOT ENABLED  ->  gcloud services enable $api"
  fi
done

echo
echo "=== GPU quota in asia-south1 ==="
gcloud compute regions describe asia-south1 \
  --format="table(quotas.metric,quotas.limit,quotas.usage)" 2>/dev/null \
  | grep -E "NVIDIA_L4_GPUS|NVIDIA_T4_GPUS|G2_CPUS|PREEMPTIBLE_NVIDIA_L4_GPUS|^METRIC" \
  || echo "  could not read region quotas"

echo
echo "=== GLOBAL GPU quota (separate from regional - this one blocked us) ==="
gcloud compute project-info describe --project="$PROJ" --format="json(quotas)" 2>/dev/null \
| python3 -c "
import sys,json
for q in json.load(sys.stdin).get('quotas',[]):
    if 'GPU' in q['metric']:
        flag = '  <-- MUST BE >= 1' if q['limit'] < 1 else '  OK'
        print(f\"  {q['metric']:<24}{q['limit']:>6.0f}{flag}\")
" || echo "  could not read project quotas"

echo
echo "=== budget alerts (STRONGLY recommended before provisioning) ==="
gcloud beta billing budgets list --billing-account="$(gcloud beta billing projects describe "$PROJ" --format='value(billingAccountName)' 2>/dev/null | sed 's|.*/||')" \
  --format="table(displayName,amount.specifiedAmount.units)" 2>/dev/null \
  || echo "  none found or no permission — set one in the console before you provision"
