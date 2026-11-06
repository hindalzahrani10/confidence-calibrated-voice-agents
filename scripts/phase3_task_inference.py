"""Phase 3: ASR quality, structured task inference and calibration.

Calibration (one temperature on the lexical belief model) is fitted on TRAIN only and checked on VAL.
The held-out TEST split is excluded unless --include-test is passed AND reports/protocol_freeze.json exists,
so nothing is tuned or inspected on it during development.

    python scripts/phase3_task_inference.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src import evalutil as E  # noqa: E402
from src.ontology import BeliefModel, Task, load_catalogue, observe  # noqa: E402

SNAP = ROOT / "data/phase1/snapshots/phase1.v2.0+05471eebec39"
BUCKET_ORDER = ["clean", "20", "10", "5"]


def load_inputs():
    recs = {r["base_id"]: r for r in map(json.loads, (SNAP / "base_items.jsonl").read_text().splitlines())}
    man = [json.loads(l) for l in (ROOT / "data/phase2/audio.v1/audio_manifest.jsonl").read_text().splitlines()]
    asr = {json.loads(l)["clip_id"]: json.loads(l) for l in (ROOT / "data/phase3/asr.v1/asr_results.jsonl").read_text().splitlines()}
    init = [m for m in man if m["kind"] == "initial" and m["clip_id"] in asr]
    return recs, init, asr


def bucket(m) -> str:
    return "clean" if m["snr_nominal_db"] is None else str(m["snr_nominal_db"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--include-test", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "reports/phase3"))
    a = ap.parse_args()
    splits = ["train", "val"]
    if a.include_test:
        if not (ROOT / "reports/protocol_freeze.json").exists():
            raise SystemExit("refusing to touch the test split: reports/protocol_freeze.json does not exist")
        splits.append("test")
    out = Path(a.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load((ROOT / "configs/default.yaml").read_text())

    recs, init, asr = load_inputs()
    todo = [m for m in init if m["split"] in splits]
    all_init = [json.loads(l) for l in (ROOT / "data/phase2/audio.v1/audio_manifest.jsonl").read_text().splitlines()]
    expected = sum(1 for m in all_init if m["kind"] == "initial" and m["split"] in splits)
    if len(todo) < expected:
        raise SystemExit(f"initial-utterance ASR incomplete: {len(todo)}/{expected} clips; wait for scripts/run_asr.py")
    cat = load_catalogue(ROOT / "data/phase3/catalogue.v1.json")
    bm = BeliefModel(cat)

    rows = []
    for m in todo:
        r = recs[m["owner_base_id"]]
        P = r["private"]
        rec = asr[m["clip_id"]]
        S = bm.scores(observe(rec["words"]), tie_key=m["clip_id"])
        acc = E.accepted_mask(cat, P["acceptable_tasks"])
        rows.append(dict(clip=m["clip_id"], base=r["base_id"], split=m["split"], cat=P["ambiguity_type"], bucket=bucket(m),
                         S=S, acc=acc, ref=P["reference_text"], hyp=rec["text"], gold=P["gold_task"], pats=P["acceptable_tasks"],
                         runtime=rec["runtime_s"]))

    # ----- ASR quality (descriptive) --------------------------------------------------------------------
    def agg(sel, fn):
        e = n = 0
        for x in sel:
            a_, b_ = fn(x["ref"], x["hyp"])
            e += a_; n += b_
        return e / n if n else float("nan")

    asr_tab = {"by_snr": {}, "by_category": {}, "by_snr_category": {}}
    for sp in splits:
        for b in BUCKET_ORDER:
            sel = [x for x in rows if x["split"] == sp and x["bucket"] == b]
            asr_tab["by_snr"][f"{sp}/{b}"] = {"n": len(sel), "WER": agg(sel, E.wer), "CER": agg(sel, E.cer),
                                              "mean_runtime_s": float(np.mean([x["runtime"] for x in sel]))}
    for sp in splits:
        for c in ("none", "acoustic", "intent", "underspecified"):
            sel = [x for x in rows if x["split"] == sp and x["cat"] == c]
            asr_tab["by_category"][f"{sp}/{c}"] = {"n": len(sel), "WER": agg(sel, E.wer), "CER": agg(sel, E.cer)}
            for b in BUCKET_ORDER:
                s2 = [x for x in sel if x["bucket"] == b]
                asr_tab["by_snr_category"][f"{sp}/{c}/{b}"] = {"n": len(s2), "WER": agg(s2, E.wer)}

    # ----- calibration: fit temperature on TRAIN only ------------------------------------------------------
    def post(x, beta):
        z = beta * x["S"]; z = z - z.max(); p = np.exp(z); return p / p.sum()

    def nll_split(sp, beta):
        return E.nll(np.array([post(x, beta)[x["acc"]].sum() for x in rows if x["split"] == sp]))

    grid = np.exp(np.linspace(np.log(0.02), np.log(20), 120))
    nlls = [nll_split("train", g) for g in grid]
    beta = float(grid[int(np.argmin(nlls))])
    calib = {"beta": beta, "fit_on": "train", "n_fit_variants": sum(x["split"] == "train" for x in rows),
             "train_nll_at_beta": min(nlls), "grid": [float(grid[0]), float(grid[-1])],
             "asr_results_sha256": hashlib.sha256((ROOT / "data/phase3/asr.v1/asr_results.jsonl").read_bytes()).hexdigest()[:16],
             "grammar_sha256": hashlib.sha256((ROOT / "src/ontology.py").read_bytes()).hexdigest()[:16]}
    (ROOT / "data/phase3/calibration.v1.json").write_text(json.dumps(calib, indent=2) + "\n")

    # ----- task inference metrics ---------------------------------------------------------------------------------
    def metrics(sel, beta_):
        if not sel:
            return {}
        conf, corr, mass, intent_ok, slot_ok = [], [], [], [], []
        for x in sel:
            p = post(x, beta_)
            top = int(p.argmax())
            t = cat[top]
            conf.append(float(p[top])); corr.append(float(x["acc"][top])); mass.append(float(p[x["acc"]].sum()))
            intent_ok.append(float(any(t.intent == q["intent"] for q in x["pats"])))
            gs = x["gold"]["slots"]
            slot_ok.append(float(np.mean([t.sdict.get(k) == v for k, v in gs.items()])) if gs else float("nan"))
        conf, corr, mass = map(np.array, (conf, corr, mass))
        return {"n": len(sel), "joint_acc": float(corr.mean()), "intent_acc": float(np.mean(intent_ok)),
                "slot_acc": float(np.nanmean(slot_ok)), "ECE": E.ece(conf, corr, cfg["eval"]["ece_bins"]),
                "Brier": E.brier(conf, corr), "NLL": E.nll(mass), "mean_conf": float(conf.mean())}

    inf = {}
    for sp in splits:
        sel = [x for x in rows if x["split"] == sp]
        inf[sp] = {"calibrated": metrics(sel, beta), "uncalibrated_beta1": metrics(sel, 1.0)}
        inf[sp]["by_snr"] = {b: metrics([x for x in sel if x["bucket"] == b], beta) for b in BUCKET_ORDER}
        inf[sp]["by_category"] = {c: metrics([x for x in sel if x["cat"] == c], beta)
                                  for c in ("none", "acoustic", "intent", "underspecified")}

    # confusion patterns (train+val, calibrated MAP vs accepted)
    from src.ontology import PATTERNS
    conf_ctr = Counter()
    for x in rows:
        top = int(post(x, beta).argmax())
        if not x["acc"][top]:
            g = Task.make(x["gold"]["intent"], x["gold"]["slots"])
            conf_ctr[(g.render() if g.intent in PATTERNS else g.intent, cat[top].render())] += 1
    result = {"splits": splits, "calibration": calib, "asr": asr_tab, "task_inference": inf,
              "top_confusions": [{"gold": g, "predicted": p, "count": c} for (g, p), c in conf_ctr.most_common(12)],
              "note": "Gold transcripts and labels are used for evaluation only. TEST excluded unless frozen protocol exists."}
    (out / "phase3_results.json").write_text(json.dumps(result, indent=2, default=float) + "\n")

    # ----- figures ------------------------------------------------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    for sp, mk in (("train", "o-"), ("val", "s--")):
        if sp in splits:
            ax[0].plot(range(4), [asr_tab["by_snr"][f"{sp}/{b}"]["WER"] for b in BUCKET_ORDER], mk, label=sp)
    ax[0].set_xticks(range(4)); ax[0].set_xticklabels(["clean", "20 dB", "10 dB", "5 dB"]); ax[0].set_ylabel("WER")
    ax[0].set_title("ASR word error rate vs SNR (synthetic TTS speech)"); ax[0].legend()
    for sp, mk in (("train", "o-"), ("val", "s--")):
        if sp in splits:
            for tag, bb, col in (("calibrated", beta, "C0"), ("uncalibrated", 1.0, "C1")):
                sel = [x for x in rows if x["split"] == sp]
                conf = np.array([post(x, bb).max() for x in sel]); corr = np.array([float(x["acc"][int(post(x, bb).argmax())]) for x in sel])
                rel = E.reliability(conf, corr, 6)
                if rel:
                    ax[1].plot([r[0] for r in rel], [r[1] for r in rel], mk, color=col, alpha=0.8, label=f"{sp} {tag}")
    ax[1].plot([0, 1], [0, 1], "k:"); ax[1].set_xlabel("confidence (max task probability)"); ax[1].set_ylabel("accuracy")
    ax[1].set_title("Reliability of task belief"); ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / "figures/phase3_asr_and_calibration.png", dpi=160)

    print(json.dumps({"beta": beta, "n": len(rows)}, default=float))
    for sp in splits:
        print(sp, json.dumps(inf[sp]["calibrated"], default=float))
    print("WER by SNR (train):", {b: round(asr_tab["by_snr"][f"train/{b}"]["WER"], 3) for b in BUCKET_ORDER})
    return 0


if __name__ == "__main__":
    sys.exit(main())
