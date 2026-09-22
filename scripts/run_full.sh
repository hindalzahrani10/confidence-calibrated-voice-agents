#!/usr/bin/env bash
# Full sweep over the frozen evaluation set.
# Only run this once the pilot is clean and the eval set is frozen.
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="${1:-configs/default.yaml}"

python scripts/build_data.py --config "$CONFIG"
python -m src.run_agent --config "$CONFIG"
python -m src.evaluate --config "$CONFIG"

echo
echo "Done. results/policy_comparison.csv fills table 1 of the paper;"
echo "results/signal_calibration.csv fills the ablation grid."
