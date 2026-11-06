"""Tune the two threshold baselines on VALIDATION episodes only (maximise mean utility; ties -> lower threshold).

NM-VoI, EVPI and the ablations have no tuned hyperparameters: costs are fixed a priori (src/nmvoi.py Costs),
the channel is fitted on train, the temperature on train.  Writes data/phase5/tuned.json.
"""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import run_episodes as R  # noqa: E402
from src import nmvoi as V  # noqa: E402

w = R.World("mixed")
vals = w.variants(["val"])
res = {"fit_on": "val", "n_variants": len(vals), "tau": {}, "h": {}}
best = {}
for name, grid, mk in (("tau", [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95], V.ConfidenceThreshold),
                       ("h", [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0], V.EntropyThreshold)):
    rows = []
    for g in grid:
        eps = [R.run_episode(w, mk(g), m) for m in vals]
        if any(e["evaluator"]["status"] != "complete" for e in eps):
            w.replies.flush_requests()
            raise SystemExit(f"pending replies while tuning {name}={g}: run scripts/generate_ondemand.py, then re-run")
        s = R.summarise(eps)
        rows.append((g, s["utility"], s["success"], s["clarification_rate"]))
        res[name][str(g)] = {"utility": s["utility"], "success": s["success"], "ask_rate": s["clarification_rate"]}
    best[name] = max(rows, key=lambda r: (round(r[1], 9), -r[0]))[0]
res["selected"] = best
out = ROOT / "data/phase5"; out.mkdir(exist_ok=True)
(out / "tuned.json").write_text(json.dumps({"tau": best["tau"], "h": best["h"], "detail": res}, indent=1) + "\n")
print("selected on val:", best)
