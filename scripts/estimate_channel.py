"""Phase 4.1: estimate the clarification reply channel from DEV data only.

For every TRAIN item and every SNR bucket, each Phase 1 reply clip (yes / no / selected option / none of
those / open answer) was recognised by ASR and parsed by the agent-side parsers. The empirical recognition
rate per (question type, SNR bucket) is r_q(SNR) in notes/Formulation.md §4.1. Val items are used only to
check how well the train-based estimate predicts held-out recognition (no refitting).

    python scripts/estimate_channel.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import evalutil as E  # noqa: E402
from src.channel import BUCKETS, ChannelTable, parse_closed, parse_confirm, wilson  # noqa: E402
from src.ontology import BeliefModel, load_catalogue, observe  # noqa: E402

SNAP = ROOT / "data/phase1/snapshots/phase1.v2.0+05471eebec39"
M_SUPPORT = 8


def collect(splits, recs, man, asr, cat, bm, beta):
    counts = {k: {b: [0, 0] for b in BUCKETS} for k in ("confirm_yes", "confirm_no", "closed", "open")}
    by_clip = {(m["owner_base_id"], m["reply_key"], str(m["snr_nominal_db"])): m for m in man if m["kind"] == "reply"}
    init = {(m["owner_base_id"], str(m["snr_nominal_db"])): m for m in man if m["kind"] == "initial"}
    skipped = 0
    for r in recs.values():
        if r["split"] not in splits:
            continue
        P = r["private"]
        opens = [p for p in P["probes"] if p["question_type"] == "open"]
        open_probe = max(opens, key=lambda p: len(p["covers"]), default=None)
        for p in P["probes"]:
            for snr in (None, 20, 10, 5):
                b = "clean" if snr is None else str(snr)
                text = p["reply"]["text"].strip()
                import hashlib
                rkey = hashlib.sha256(f"{r['base_id']}|{text.lower()}".encode()).hexdigest()[:8]
                m = by_clip.get((r["base_id"], rkey, str(snr)))
                if m is None or m["clip_id"] not in asr:
                    skipped += 1
                    continue
                said = asr[m["clip_id"]]
                rt = p["reply"]["type"]
                if rt == "yes":
                    k, ok = "confirm_yes", parse_confirm(said["text"])[0] == "yes"
                elif rt in ("no_bare", "no_correction"):
                    k, ok = "confirm_no", parse_confirm(said["text"])[0] == "no"
                elif rt == "select":
                    gold_idx = p["option_values"].index(P["gold_task"]["slots"].get(p["covers"][0], P["gold_task"]["intent"]))
                    k, ok = "closed", parse_closed(said["text"], p["options"]) == str(gold_idx)
                elif rt == "none_of_these":
                    k, ok = "closed", parse_closed(said["text"], p["options"]) == "none"
                elif rt == "answer":
                    if p is not open_probe:
                        continue
                    im = init[(r["base_id"], str(snr))]
                    if im["clip_id"] not in asr:
                        skipped += 1
                        continue
                    S0 = bm.scores(observe(asr[im["clip_id"]]["words"]), tie_key=im["clip_id"])
                    b0 = np.exp(beta * (S0 - S0.max())); b0 /= b0.sum()
                    support = list(np.argsort(-b0)[:M_SUPPORT])
                    S1 = bm.scores(observe(asr[im["clip_id"]]["words"]) + observe(said["words"]), tie_key=im["clip_id"])
                    yhat = max(support, key=lambda y: S1[y])
                    acc = E.accepted_mask(cat, P["acceptable_tasks"])
                    k, ok = "open", bool(acc[yhat])
                else:
                    continue
                counts[k][b][0] += int(ok)
                counts[k][b][1] += 1
    return counts, skipped


def main() -> int:
    recs = {r["base_id"]: r for r in map(json.loads, (SNAP / "base_items.jsonl").read_text().splitlines())}
    man = [json.loads(l) for l in (ROOT / "data/phase2/audio.v1/audio_manifest.jsonl").read_text().splitlines()]
    asr = {json.loads(l)["clip_id"]: json.loads(l) for l in (ROOT / "data/phase3/asr.v1/asr_results.jsonl").read_text().splitlines()}
    cat = load_catalogue(ROOT / "data/phase3/catalogue.v1.json")
    beta = json.loads((ROOT / "data/phase3/calibration.v1.json").read_text())["beta"]
    bm = BeliefModel(cat, beta)
    tr, sk_tr = collect({"train"}, recs, man, asr, cat, bm, beta)
    va, sk_va = collect({"val"}, recs, man, asr, cat, bm, beta)
    if sk_tr:
        raise SystemExit(f"{sk_tr} train reply clips lack ASR output; wait for scripts/run_asr.py to finish")
    table = ChannelTable(tr)
    out = {"fit_on": "train", "buckets": BUCKETS, "prior_strength": table.prior_strength,
           "train_counts": tr, "val_counts": va, "val_skipped": sk_va, "table": {}, "val_check": {}}
    for q in tr:
        out["table"][q] = {}
        for b in BUCKETS:
            k, n = tr[q][b]
            lo, hi = wilson(k, n)
            out["table"][q][b] = {"k": k, "n": n, "raw": k / n if n else None, "wilson95": [lo, hi],
                                  "smoothed_snr": table.r(q, b, "snr"), "pooled": table.r(q, b, "pooled")}
            kv, nv = va[q][b]
            out["val_check"][f"{q}/{b}"] = {"val_k": kv, "val_n": nv, "val_rate": kv / nv if nv else None,
                                            "predicted": table.r(q, b, "snr")}
    (ROOT / "data/phase4").mkdir(exist_ok=True)
    (ROOT / "data/phase4/channel.v1.json").write_text(json.dumps({**table.to_json(), "estimate": out}, indent=1) + "\n")
    print(f"{'qtype':12}" + "".join(f"{b:>16}" for b in BUCKETS))
    for q in tr:
        print(f"{q:12}" + "".join(f"{out['table'][q][b]['k']:>4}/{out['table'][q][b]['n']:<3}={out['table'][q][b]['smoothed_snr']:.2f} " for b in BUCKETS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
