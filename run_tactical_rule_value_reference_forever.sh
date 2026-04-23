#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/yphong/omok_deeplearning"
PYTHON="$ROOT/.venv/bin/python"
LOCK_FILE="$ROOT/.locks/training_forever.lock"
cd "$ROOT"

mkdir -p "$ROOT/.locks"
exec 9>"$LOCK_FILE"
flock -n 9 || exit 0

while pgrep -f "$PYTHON train_value_reference.py" >/dev/null; do
  sleep 30
done

while true; do
  latest_version=$(
    "$PYTHON" - <<'PY'
from pathlib import Path
import re

models = Path("models")
reference_directory = models / "refer"
versions = []
for path in models.glob("tactical_rule_value_agent_v*.pt"):
    if path.stem.endswith("_reference"):
        continue
    match = re.fullmatch(r"tactical_rule_value_agent_v(\d+)", path.stem)
    if not match:
        continue
    version = int(match.group(1))
    if version >= 900:
        continue
    if (reference_directory / f"{path.stem}_reference.pt").exists():
        versions.append(version)
print(max(versions) if versions else 9)
PY
  )

  next_version=$((latest_version + 1))
  candidate_init="models/tactical_rule_value_agent_v${latest_version}.pt"

  "$PYTHON" train_value_reference.py \
    --games 1000 \
    --save-every 1000 \
    --pretrain-positions 0 \
    --candidate-prefix tactical_rule_value_agent \
    --candidate-version "$next_version" \
    --candidate-init-model "$candidate_init" \
    --reference-cycle-length 10 \
    --reference-rule-agent-level hard \
    --reference-rule-opening-moves 20 \
    --reference-rule-followup-probability 0.10 \
    --reference-rule-only-agent-level hard \
    --teacher-rule-agent-level hard \
    --teacher-weight 1.0 \
    --device cpu

  sleep 5
done
