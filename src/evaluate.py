"""Turn the run log into the paper's tables.

Reads `results/runs.jsonl` and produces:

  * results/signal_calibration.csv : one row per confidence signal
                                     (feeds the ablation grid, figure 2b)
  * results/policy_comparison.csv  : one row per policy
                                     (feeds table 1 and figure 2a)

Thresholds and the fusion model are fitted on the validation split only;
every reported number comes from the frozen test split.

Usage:
    python -m src.evaluate --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .config import load_config
from .metrics import (
    bootstrap_ci,
    clarification_metrics,
    summarise_signal,
    task_success_rate,
)
from .policies import CalibratedFusion, build_policies
from .signals import SIGNAL_NAMES, ConfidenceSignals


def load_runs(path: Path) -> List[Dict]:
    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def to_signals(record: Dict) -> ConfidenceSignals:
    """Rebuild the dataclass, ignoring any extra keys added later."""
    fields = ConfidenceSignals.__dataclass_fields__
    payload = {k: v for k, v in record["signals"].items() if k in fields}
    return ConfidenceSignals(**payload)


def evaluate_signals(records: List[Dict], n_bins: int) -> pd.DataFrame:
    """Calibration of each signal against direct-answer correctness.

    Reported per acoustic condition, because the project's central question
    is whether the best signal *changes* as the audio degrades.
    """
    rows = []
    for signal_name in SIGNAL_NAMES:
        for condition, subset in group_by_condition(records).items():
            if not subset:
                continue
            confidences = [r["signals"][signal_name] for r in subset]
            correct = [r["direct_correct"] for r in subset]
            row = {"signal": signal_name, "condition": condition}
            row.update(summarise_signal(confidences, correct, n_bins=n_bins))
            rows.append(row)
    return pd.DataFrame(rows)


def group_by_condition(records: List[Dict]) -> Dict[str, List[Dict]]:
    """Split records into clean / SNR buckets, plus an "all" bucket."""
    groups: Dict[str, List[Dict]] = {"all": list(records)}
    for r in records:
        snr = r.get("snr_db")
        key = "clean" if snr is None else f"snr{int(snr)}"
        groups.setdefault(key, []).append(r)
    return groups


def evaluate_policies(records: List[Dict], cfg) -> pd.DataFrame:
    """Compare every active policy on the frozen test split."""
    val = [r for r in records if r["split"] == "val"]
    test = [r for r in records if r["split"] == "test"]

    if not test:
        raise SystemExit(
            "No test-split records found. Did scripts/build_data.py assign splits?"
        )

    policies = build_policies(cfg)

    # Fit the fusion policy on validation only.
    fusion = policies.get("calibrated_fusion")
    if isinstance(fusion, CalibratedFusion):
        if not val:
            print("[warn] no validation records; fusion policy will abstain")
        else:
            fusion.fit(
                [to_signals(r) for r in val],
                [r["clarification_helps"] for r in val],
            )
            print("Fusion coefficients:", fusion.coefficients())

    rows = []
    for name, policy in policies.items():
        asked, correct, needed, turns = [], [], [], []
        for r in test:
            ask = bool(policy(to_signals(r)))
            asked.append(ask)
            correct.append(r["clarified_correct"] if ask else r["direct_correct"])
            needed.append(r["ambiguity_type"] != "none")
            turns.append(2 if ask else 1)

        mean_success, lo, hi = bootstrap_ci(
            [float(c) for c in correct], n_bootstrap=cfg["eval"]["n_bootstrap"]
        )

        row = {
            "policy": name,
            "task_success": task_success_rate(correct),
            "success_ci_low": lo,
            "success_ci_high": hi,
            "mean_turns": sum(turns) / len(turns),
        }
        row.update(clarification_metrics(asked, needed))
        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--runs", default=None, help="Path to runs.jsonl")
    args = parser.parse_args()

    cfg = load_config(args.config)
    results_dir = cfg.path("results_dir")
    runs_path = Path(args.runs) if args.runs else results_dir / "runs.jsonl"

    if not runs_path.exists():
        raise SystemExit(f"{runs_path} not found. Run `python -m src.run_agent` first.")

    records = load_runs(runs_path)
    print(f"Loaded {len(records)} records from {runs_path}")

    signal_df = evaluate_signals(records, n_bins=cfg["eval"]["ece_bins"])
    signal_path = results_dir / "signal_calibration.csv"
    signal_df.to_csv(signal_path, index=False)
    print(f"\nWrote {signal_path}")
    print(signal_df.to_string(index=False))

    policy_df = evaluate_policies(records, cfg)
    policy_path = results_dir / "policy_comparison.csv"
    policy_df.to_csv(policy_path, index=False)
    print(f"\nWrote {policy_path}")
    print(policy_df.to_string(index=False))

    print(
        "\nThese two CSVs map directly onto the paper: policy_comparison.csv "
        "fills table 1, signal_calibration.csv fills the ablation grid."
    )


if __name__ == "__main__":
    main()
