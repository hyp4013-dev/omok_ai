#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/yphong/omok_deeplearning"
PYTHON="$ROOT/.venv/bin/python"
LOCK_FILE="$ROOT/.locks/training_forever.lock"
cd "$ROOT"

mkdir -p "$ROOT/.locks"
exec 9>"$LOCK_FILE"
flock -n 9 || exit 0

while pgrep -f "$PYTHON train_hybrid_mix_reference.py" >/dev/null; do
  sleep 30
done

while true; do
  latest_version=$(
    "$PYTHON" - <<'PY'
from pathlib import Path
import re
import torch

models = Path("models")
candidate_latest = models / "tactical_rule_hybrid_mix_weight_agent_candidate_latest.pt"
if candidate_latest.exists():
    try:
        payload = torch.load(candidate_latest, map_location="cpu")
        name = payload.get("name", "")
        match = re.fullmatch(r"tactical_rule_hybrid_mix_weight_agent_v(\d+)", name)
        if match:
            print(int(match.group(1)))
            raise SystemExit
    except Exception:
        pass

versions = []
for path in models.glob("tactical_rule_hybrid_mix_weight_agent_v*.pt"):
    match = re.fullmatch(r"tactical_rule_hybrid_mix_weight_agent_v(\d+)", path.stem)
    if match:
        versions.append(int(match.group(1)))
print(max(versions) if versions else 0)
PY
  )

  next_version=$((latest_version + 1))
  candidate_init=""
  if [ "$latest_version" -gt 0 ]; then
    candidate_init="--candidate-init-model models/tactical_rule_hybrid_mix_weight_agent_v${latest_version}.pt"
  else
    candidate_init="--candidate-init-model models/tactical_rule_hybrid_agent_v6.pt"
  fi

  # shellcheck disable=SC2086
  "$PYTHON" train_hybrid_mix_reference.py \
    --games 1000 \
    --save-every 1000 \
    --candidate-prefix tactical_rule_hybrid_mix_weight_agent \
    --candidate-version "$next_version" \
    $candidate_init \
    --reference-cycle-length 10 \
    --reference-rule-agent-level hard \
    --reference-rule-opening-moves 20 \
    --reference-rule-followup-probability 0.10 \
    --reference-rule-only-agent-level hard \
    --policy-mix-weight 0.01 \
    --policy-aux-weight 0.05 \
    --device cpu

  sleep 5
done
