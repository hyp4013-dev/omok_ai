#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/yphong/omok_deeplearning"
PYTHON="$ROOT/.venv/bin/python"
LOCK_FILE="$ROOT/.locks/training_forever.lock"
cd "$ROOT"

mkdir -p "$ROOT/.locks"
exec 9>"$LOCK_FILE"
flock -n 9 || exit 0

while pgrep -f "$PYTHON train_policy_only_reference.py" >/dev/null; do
  sleep 30
done

while true; do
  latest_version=$(
    "$PYTHON" - <<'PY'
from pathlib import Path
import re
import torch

models = Path("models")
candidate_latest = models / "tactical_rule_policy_agent_candidate_latest.pt"
if candidate_latest.exists():
    try:
        payload = torch.load(candidate_latest, map_location="cpu")
        name = payload.get("name", "")
        match = re.fullmatch(r"tactical_rule_policy_agent_v(\d+)", name)
        if match:
            print(int(match.group(1)))
            raise SystemExit
    except Exception:
        pass

versions = []
for path in models.glob("tactical_rule_policy_agent_v*.pt"):
    match = re.fullmatch(r"tactical_rule_policy_agent_v(\d+)", path.stem)
    if match:
        versions.append(int(match.group(1)))
print(max(versions) if versions else 0)
PY
  )

  next_version=$((latest_version + 1))
  candidate_init=""
  candidate_feature_init=""
  if [ "$latest_version" -gt 0 ]; then
    candidate_init="--candidate-init-model models/tactical_rule_policy_agent_v${latest_version}.pt"
  else
    candidate_feature_init="--candidate-feature-init-model models/tactical_rule_hybrid_agent_v37.pt"
  fi

  # shellcheck disable=SC2086
  "$PYTHON" train_policy_only_reference.py \
    --games 1000 \
    --save-every 1000 \
    --candidate-prefix tactical_rule_policy_agent \
    --candidate-version "$next_version" \
    $candidate_init \
    $candidate_feature_init \
    --reference-rule-agent-level super_easy \
    --reference-rule-opening-moves 20 \
    --reference-rule-followup-probability 0.10 \
    --teacher-rule-agent-level hard \
    --teacher-weight 1.0 \
    --opening-teacher-moves 20 \
    --device cpu

  sleep 5
done
