#!/usr/bin/env bash
# Pilot run: 20 items, end to end. Do this FIRST.
# A 20-item pilot catches almost every pipeline bug for a fraction of the
# compute of a full sweep.
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="${1:-configs/default.yaml}"

echo "==> Building evaluation set"
python scripts/build_data.py --config "$CONFIG"

echo "==> Running the agent on 20 items"
python -m src.run_agent --config "$CONFIG" --limit 20

echo "==> Scoring"
python -m src.evaluate --config "$CONFIG"

echo
echo "Pilot complete. Check results/runs.jsonl and confirm that:"
echo "  1. every record has all five confidence signals"
echo "  2. transcripts look sane"
echo "  3. both branches (direct and clarified) produced answers"
