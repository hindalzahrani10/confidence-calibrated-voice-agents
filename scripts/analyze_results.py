"""Tables, paired statistics and figures from episode logs (evaluator side).

Uncertainty respects dependence: the four SNR variants of one base item form a cluster, and every interval
resamples whole base items (never variants or noise draws as independent units). Paired comparisons use the
same clusters under both policies.

    python scripts/analyze_results.py --episodes data/phase5/dev.v1/episodes.jsonl --tag dev
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BUCKETS = ["clean", "20", "10", "5"]
MAIN = ["never_ask", "always_ask", "confidence_threshold", "entropy_threshold", "evpi", "noise_indep_voi", "nm_voi"]
ABL = ["nm_voi", "abl_equal_costs", "abl_open_only", "abl_closed_only", "abl_confirm_only", "abl_no_abstain",
       "abl_uncalibrated", "noise_indep_voi", "evpi"]
METRICS = ["success", "wrong", "abstained", "asked", "turns", "cost", "utility", "unnecessary_clarification"]


def load(path):
    eps = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    pend = sum(e["evaluator"]["status"] != "complete" for e in eps)
    return [e for e in eps if e["evaluator"]["status"] == "complete"], pend


def table(eps, policy, key, sel=lambda e: True):
    d = defaultdict(list)
    for e in eps:
        if e["agent"]["policy"] == policy and sel(e):
            d[e["evaluator"]["base_id"]].append(float(e["evaluator"][key]))
    return {k: np.array(v) for k, v in d.items()}


def boot(clusters: dict, B=2000, seed=0):
    ks = sorted(clusters)
    if not ks:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    point = float(np.concatenate([clusters[k] for k in ks]).mean())
    bs = [float(np.concatenate([clusters[ks[i]] for i in rng.integers(0, len(ks), len(ks))]).mean()) for _ in range(B)]
    return point, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def paired(a: dict, b: dict, B=2000, seed=0):
    ks = sorted(set(a) & set(b))
    if not ks:
        return (float("nan"),) * 3
    d = {k: a[k] - b[k] for k in ks}
    return boot(d, B, seed)


def fmt(t):
    return f"{t[0]:.3f} [{t[1]:.3f}, {t[2]:.3f}]"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--splits", nargs="*", default=None, help="restrict to these splits (default: all in file)")
    a = ap.parse_args()
    eps, pend = load(a.episodes)
    if a.splits:
        eps = [e for e in eps if e["evaluator"]["split"] in a.splits]
    out = ROOT / "reports/results" / a.tag
    (out / "figures").mkdir(parents=True, exist_ok=True)
    pols = [p for p in dict.fromkeys(e["agent"]["policy"] for e in eps)]
    n_items = len({e["evaluator"]["base_id"] for e in eps})
    lines = [f"# Results ({a.tag})", "", f"Base items: {n_items}; episodes (complete): {len(eps)}; pending/excluded: {pend}.",
             "Intervals are 95% percentile bootstrap resampling whole base items (4 SNR variants per item are one cluster).",
             "Utility = reward (+1 correct, -1 wrong, 0 abstain) minus interaction cost (open 0.30, closed 0.20, confirm 0.10).", ""]
    summary = {}

    # --- main table ---------------------------------------------------------------------------------------
    lines += ["## Policy comparison (all SNRs pooled)", "",
              "| policy | success | wrong | abstain | ask rate | turns | cost | utility | unnecessary clar. |", "|---|---|---|---|---|---|---|---|---|"]
    shown = []
    for p in MAIN + [x for x in ABL if x not in MAIN]:
        if p not in pols or p in shown:
            continue
        shown.append(p)
        row = {m: boot(table(eps, p, m)) for m in METRICS}
        summary[p] = {m: {"mean": v[0], "ci": [v[1], v[2]]} for m, v in row.items()}
        lines.append(f"| {p} | " + " | ".join(f"{row[m][0]:.3f} [{row[m][1]:.2f},{row[m][2]:.2f}]" for m in
                                              ["success", "wrong", "abstained", "asked", "turns", "cost", "utility", "unnecessary_clarification"]) + " |")
    with open(out / "main_table.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["policy"] + [f"{m}_{s}" for m in METRICS for s in ("mean", "lo", "hi")])
        for p, d in summary.items():
            wr.writerow([p] + [x for m in METRICS for x in (d[m]["mean"], *d[m]["ci"])])

    # --- paired comparisons vs nm_voi ------------------------------------------------------------------------
    lines += ["", "## Paired differences: nm_voi minus other (positive favours nm_voi)", "",
              "| vs | d utility | d success | d turns | d wrong |", "|---|---|---|---|---|"]
    paired_out = {}
    if "nm_voi" in pols:
        for p in [x for x in shown if x != "nm_voi"]:
            r = {m: paired(table(eps, "nm_voi", m), table(eps, p, m)) for m in ("utility", "success", "turns", "wrong")}
            paired_out[p] = {m: {"diff": v[0], "ci": [v[1], v[2]]} for m, v in r.items()}
            lines.append(f"| {p} | " + " | ".join(fmt(r[m]) for m in ("utility", "success", "turns", "wrong")) + " |")

    # --- by SNR (true condition) --------------------------------------------------------------------------------------
    snr_of = lambda e: "clean" if e["evaluator"]["snr_nominal_db"] is None else str(e["evaluator"]["snr_nominal_db"])  # noqa: E731
    bysnr = {}
    lines += ["", "## By SNR (utility / success / ask rate)", "", "| policy | " + " | ".join(BUCKETS) + " |", "|---|---|---|---|---|"]
    for p in shown:
        cells = []
        for b in BUCKETS:
            u = boot(table(eps, p, "utility", lambda e, b=b: snr_of(e) == b)); s = boot(table(eps, p, "success", lambda e, b=b: snr_of(e) == b))
            k = boot(table(eps, p, "asked", lambda e, b=b: snr_of(e) == b))
            bysnr[(p, b)] = (u[0], s[0], k[0])
            cells.append(f"{u[0]:+.2f} / {s[0]:.2f} / {k[0]:.2f}")
        lines.append(f"| {p} | " + " | ".join(cells) + " |")

    # --- by category ------------------------------------------------------------------------------------------------------
    cats = ["none", "acoustic", "intent", "underspecified"]
    bycat = {}
    lines += ["", "## By ambiguity category (utility / success)", "", "| policy | " + " | ".join(cats) + " |", "|---|---|---|---|---|"]
    for p in shown:
        cells = []
        for c in cats:
            u = boot(table(eps, p, "utility", lambda e, c=c: e["evaluator"]["category"] == c)); s = boot(table(eps, p, "success", lambda e, c=c: e["evaluator"]["category"] == c))
            bycat[(p, c)] = (u[0], s[0])
            cells.append(f"{u[0]:+.2f} / {s[0]:.2f}")
        lines.append(f"| {p} | " + " | ".join(cells) + " |")
    n_by_cat = {c: len({e['evaluator']['base_id'] for e in eps if e['evaluator']['category'] == c}) for c in cats}
    lines += ["", f"Items per category in this set: {n_by_cat}. Per-category cells hold very few items; treat as descriptive."]

    # --- action distribution ---------------------------------------------------------------------------------------------------
    act = defaultdict(lambda: defaultdict(int))
    for e in eps:
        a_ = e["agent"]
        label = "answer" if a_["action"] == "answer" else ("abstain" if a_["action"] == "abstain" else f"ask_{a_['question']['type']}")
        act[a_["policy"]][label] += 1
    labels = ["answer", "abstain", "ask_open", "ask_closed", "ask_confirm"]
    lines += ["", "## Initial action distribution (fraction of episodes)", "", "| policy | " + " | ".join(labels) + " |", "|---|---|---|---|---|---|"]
    for p in shown:
        tot = sum(act[p].values())
        lines.append(f"| {p} | " + " | ".join(f"{act[p][l] / tot:.2f}" for l in labels) + " |")
    (out / "results.md").write_text("\n".join(lines) + "\n")
    (out / "results.json").write_text(json.dumps({"summary": summary, "paired_vs_nm_voi": paired_out,
                                                  "by_snr": {f"{p}/{b}": v for (p, b), v in bysnr.items()},
                                                  "by_category": {f"{p}/{c}": v for (p, c), v in bycat.items()},
                                                  "actions": {p: dict(d) for p, d in act.items()},
                                                  "n_items": n_items, "pending_excluded": pend}, indent=1, default=float) + "\n")

    # --- figures (all data from the episode log) -------------------------------------------------------------------------------
    def snr_plot(idx, ylabel, fname, pl):
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        for p in pl:
            if p in shown:
                ax.plot(range(4), [bysnr[(p, b)][idx] for b in BUCKETS], "o-", label=p, lw=1.6 if p == "nm_voi" else 1.0)
        ax.set_xticks(range(4)); ax.set_xticklabels(["clean", "20 dB", "10 dB", "5 dB"]); ax.set_ylabel(ylabel)
        ax.legend(fontsize=6); fig.tight_layout(); fig.savefig(out / "figures" / fname, dpi=170); plt.close(fig)
    snr_plot(1, "task success rate", "fig1_success_vs_snr.png", MAIN)
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for p in MAIN:
        if p in shown:
            ax.plot(range(4), [bysnr[(p, b)][2] for b in BUCKETS], "o-", label=p, lw=1.6 if p == "nm_voi" else 1.0)
    ax.set_xticks(range(4)); ax.set_xticklabels(["clean", "20 dB", "10 dB", "5 dB"]); ax.set_ylabel("clarification turns (= ask rate)")
    ax.legend(fontsize=6); fig.tight_layout(); fig.savefig(out / "figures/fig2_turns_vs_snr.png", dpi=170); plt.close(fig)
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    for p in shown:
        s = summary[p]
        ax.errorbar(s["cost"]["mean"], s["success"]["mean"], fmt="o", label=p, ms=7 if p == "nm_voi" else 5)
    ax.set_xlabel("mean interaction cost"); ax.set_ylabel("task success rate"); ax.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(out / "figures/fig3_success_vs_cost.png", dpi=170); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    bottom = np.zeros(len(shown))
    for l in labels:
        vals = np.array([act[p][l] / sum(act[p].values()) for p in shown])
        ax.bar(range(len(shown)), vals, bottom=bottom, label=l); bottom += vals
    ax.set_xticks(range(len(shown))); ax.set_xticklabels(shown, rotation=60, ha="right", fontsize=6); ax.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(out / "figures/fig6_action_distribution.png", dpi=170); plt.close(fig)
    ab = [p for p in ABL if p in shown]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.bar(range(len(ab)), [summary[p]["utility"]["mean"] for p in ab],
           yerr=[[summary[p]["utility"]["mean"] - summary[p]["utility"]["ci"][0] for p in ab], [summary[p]["utility"]["ci"][1] - summary[p]["utility"]["mean"] for p in ab]], capsize=2)
    ax.set_xticks(range(len(ab))); ax.set_xticklabels(ab, rotation=60, ha="right", fontsize=6); ax.set_ylabel("mean utility")
    fig.tight_layout(); fig.savefig(out / "figures/fig7_ablations.png", dpi=170); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    w_ = 0.8 / max(len(MAIN), 1)
    for i, p in enumerate([x for x in MAIN if x in shown]):
        ax.bar(np.arange(4) + i * w_, [bycat[(p, c)][0] for c in cats], w_, label=p)
    ax.set_xticks(np.arange(4) + 0.4); ax.set_xticklabels(cats); ax.set_ylabel("mean utility"); ax.legend(fontsize=5)
    fig.tight_layout(); fig.savefig(out / "figures/fig8_by_category.png", dpi=170); plt.close(fig)
    print("\n".join(lines[:30]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
