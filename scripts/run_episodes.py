"""Run dialogue episodes: agent policy x item x SNR variant -> evaluator-scored outcomes.

Strict separation:
  agent side      src.ontology / src.channel / src.nmvoi   (observations only)
  evaluator side  src.simulator / src.audio_service / src.evalutil  (hold the hidden task)
The agent receives: the ASR words of the initial audio, a blind SNR bucket from the audio samples, and
(after it asks) the ASR words of the simulated reply. Nothing else.

Two-pass design: a reply that is not yet synthesised/recognised raises MissingReply; the request is logged,
the episode is marked `pending`, and `scripts/generate_ondemand.py` fills it. Re-run to complete.

    python scripts/run_episodes.py --splits train val --policies all --out data/phase5/dev.v1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import evalutil as E  # noqa: E402
from src import nmvoi as V  # noqa: E402
from src.audio_service import MissingReply, ReplyService  # noqa: E402
from src.channel import ChannelTable, estimate_snr_bucket, parse_closed, parse_confirm  # noqa: E402
from src.ontology import BeliefModel, load_catalogue, observe  # noqa: E402
from src.simulator import UserSimulator  # noqa: E402

SNAP = ROOT / "data/phase1/snapshots/phase1.v2.0+05471eebec39"
M_SUPPORT = 8
TRUE_COSTS = V.Costs()


def question_text(q: V.Question, cat) -> str:
    if q.qtype == "confirm":
        return f"Do you want me to {cat[q.proposal].render()}?"
    if q.qtype == "closed":
        opts = list(q.option_texts) + ["something else"]
        return f"Which do you mean: {', '.join(opts[:-1])}, or {opts[-1]}?"
    return "Could you tell me more precisely what you want?"


def make_policies(ch: ChannelTable, tuned: dict, names: list) -> dict:
    C = TRUE_COSTS
    eq = V.Costs(0.2, 0.2, 0.2)
    P = {
        "never_ask": V.NeverAsk(),
        "always_ask": V.AlwaysAsk(),
        "confidence_threshold": V.ConfidenceThreshold(tuned.get("tau", 0.7)),
        "entropy_threshold": V.EntropyThreshold(tuned.get("h", 0.8)),
        "evpi": V.VoI("evpi", ch, "perfect", C),
        "noise_indep_voi": V.VoI("noise_indep_voi", ch, "pooled", C),
        "nm_voi": V.VoI("nm_voi", ch, "snr", C),
        # one-component ablations of nm_voi
        "abl_equal_costs": V.VoI("abl_equal_costs", ch, "snr", eq),
        "abl_open_only": V.VoI("abl_open_only", ch, "snr", C, allowed=("open",)),
        "abl_closed_only": V.VoI("abl_closed_only", ch, "snr", C, allowed=("closed",)),
        "abl_confirm_only": V.VoI("abl_confirm_only", ch, "snr", C, allowed=("confirm",)),
        "abl_no_abstain": V.VoI("abl_no_abstain", ch, "snr", C, allow_abstain=False),
        "abl_uncalibrated": V.VoI("abl_uncalibrated", ch, "snr", C),
    }
    P["abl_uncalibrated"].uncalibrated = True
    return P if names == ["all"] else {k: P[k] for k in names}


class World:
    """Evaluator-side bundle: data, models, simulator factory."""

    def __init__(self, style: str = "mixed", channel: str = "data/phase4/channel.v1.json") -> None:
        cfg = yaml.safe_load((ROOT / "configs/default.yaml").read_text())
        self.seed = cfg["seed"]
        self.recs = {r["base_id"]: r for r in map(json.loads, (SNAP / "base_items.jsonl").read_text().splitlines())}
        self.man = [json.loads(l) for l in (ROOT / "data/phase2/audio.v1/audio_manifest.jsonl").read_text().splitlines()]
        self.asr = {json.loads(l)["clip_id"]: json.loads(l)
                    for l in (ROOT / "data/phase3/asr.v1/asr_results.jsonl").read_text().splitlines()}
        self.cat = load_catalogue(ROOT / "data/phase3/catalogue.v1.json")
        self.beta = json.loads((ROOT / "data/phase3/calibration.v1.json").read_text())["beta"]
        self.bm = BeliefModel(self.cat, self.beta)
        self.ch = ChannelTable.from_json(json.loads((ROOT / channel).read_text()))
        self.replies = ReplyService(self.seed, [ROOT / "data/phase3/asr.v1/asr_results.jsonl",
                                                ROOT / "data/phase3/asr_ondemand.v1/asr_results.jsonl"],
                                    ROOT / "data/phase3/ondemand_requests.jsonl")
        self.style = style
        self.sims = {b: UserSimulator(r, style, self.seed) for b, r in self.recs.items()}
        self.snr_centroids = json.loads((ROOT / "data/phase4/snr_estimator.v1.json").read_text())["centroids"]
        self.accmask = {b: E.accepted_mask(self.cat, r["private"]["acceptable_tasks"]) for b, r in self.recs.items()}

    def variants(self, splits):
        return [m for m in self.man if m["kind"] == "initial" and m["split"] in splits]

    def agent_inputs(self, m):
        """What the agent gets for one recording: ASR output and the blind SNR bucket from the samples."""
        a = self.asr[m["clip_id"]]
        x, _ = sf.read(ROOT / m["path"], dtype="float32")
        return a, estimate_snr_bucket(x, self.snr_centroids)


def run_episode(w: World, pol, m, collect_missing: bool = True) -> dict:
    t0 = time.time()
    base = m["owner_base_id"]
    asr_rec, bucket = w.agent_inputs(m)                 # ---- agent-visible ----
    obs = observe(asr_rec["words"])
    beta = 1.0 if getattr(pol, "uncalibrated", False) else w.beta
    b0 = w.bm.posterior(obs, beta, tie_key=m["clip_id"])
    state = V.AgentState(w.cat, b0, bucket)
    dec = pol.decide(state)
    t_decide = time.time() - t0

    sim = w.sims[base]                                  # ---- evaluator-side from here ----
    acc = w.accmask[base]
    init_top = int(b0.argmax())
    ep = {"agent": {"policy": pol.name, "clip_id": m["clip_id"], "asr_text": asr_rec["text"], "snr_bucket_est": bucket,
                    "initial_top": [(w.cat[i].render(), round(float(b0[i]), 4)) for i in np.argsort(-b0)[:5]],
                    "initial_entropy": round(V.entropy(b0), 4), "initial_max_prob": round(float(b0.max()), 4),
                    "action": dec.kind, "eus": {k: round(v, 4) for k, v in dec.eus.items()}},
          "evaluator": {"base_id": base, "split": m["split"], "snr_nominal_db": m["snr_nominal_db"],
                        "category": w.recs[base]["private"]["ambiguity_type"],
                        "initial_correct": bool(acc[init_top]), "gold": sim.gold.render()}}
    cost, turns, b = 0.0, 0, b0
    status = "complete"
    if dec.kind == "ask":
        q = dec.question
        ep["agent"]["question"] = {"type": q.qtype, "variable": q.variable, "options": list(q.option_texts),
                                   "proposal": w.cat[q.proposal].render() if q.proposal is not None else None,
                                   "text": question_text(q, w.cat)}
        turns, cost = 1, TRUE_COSTS.of(q.qtype)
        reply_text, rtype = sim.reply(q, w.cat)
        ep["evaluator"]["simulator_reply"] = {"text": reply_text, "type": rtype}
        try:
            snr = m["snr_nominal_db"]
            rr = w.replies.get(base, m["split"], reply_text, snr)
        except MissingReply:
            status = "pending"
            rr = None
        if rr is not None:
            ep["agent"]["reply_asr_text"] = rr["text"]
            mode = getattr(pol, "mode", "perfect")
            P, _ = V.outcome_model(w.cat, b0, q, w.ch, bucket, mode, M_SUPPORT)
            o = None
            if q.qtype == "confirm":
                kind, rest = parse_confirm(rr["text"])
                o = {"yes": 0, "no": 1}.get(kind)
            elif q.qtype == "closed":
                kind = parse_closed(rr["text"], q.option_texts)
                o = len(q.option_values) if kind == "none" else (int(kind) if kind.isdigit() else None)
            else:
                support = list(np.argsort(-b0)[:M_SUPPORT])
                robs = observe(rr["words"])
                if robs:
                    S_r = w.bm.scores(robs, tie_key=m["clip_id"])
                    if max(S_r[y] for y in support) > 0.5:      # the reply must explain some words of some candidate
                        S1 = w.bm.scores(obs + robs, tie_key=m["clip_id"])
                        o = support.index(max(support, key=lambda y: S1[y]))
            if o is not None:
                b = V.update(b0, P, o)
            if q.qtype == "confirm" and o == 1 and rest:      # corrective no: extra lexical evidence from the remainder
                words = rr["words"][1:] if rr["words"] else []
                robs = observe(words)
                if robs:
                    z = w.beta * w.bm.scores(robs)
                    z -= z.max()
                    b = b * np.exp(z)
                    b = b / b.sum()
            ep["agent"]["parsed_outcome"] = o
            ep["agent"]["updated_top"] = [(w.cat[i].render(), round(float(b[i]), 4)) for i in np.argsort(-b)[:3]]
    final_kind = "answer" if dec.kind == "answer" else (pol.final(b) if status == "complete" else "pending")
    if dec.kind == "abstain":
        final_kind = "abstain"
    ans = int(b.argmax())
    if final_kind == "answer":
        reward = 1.0 if acc[ans] else -1.0
    else:
        reward = 0.0
    ep["agent"].update({"final": final_kind, "final_task": w.cat[ans].render() if final_kind == "answer" else None,
                        "final_entropy": round(V.entropy(b), 4), "decision_runtime_s": round(t_decide, 5)})
    ep["evaluator"].update({"status": status, "success": bool(final_kind == "answer" and acc[ans]),
                            "wrong": bool(final_kind == "answer" and not acc[ans]), "abstained": final_kind == "abstain",
                            "asked": dec.kind == "ask", "turns": turns, "cost": cost, "reward": reward,
                            "utility": reward - cost,
                            "unnecessary_clarification": bool(dec.kind == "ask" and acc[init_top])})
    return ep


def summarise(eps: list) -> dict:
    ev = [e["evaluator"] for e in eps if e["evaluator"]["status"] == "complete"]
    if not ev:
        return {}
    g = lambda k: float(np.mean([x[k] for x in ev]))  # noqa: E731
    asked = [x for x in ev if x["asked"]]
    return {"n": len(ev), "success": g("success"), "wrong": g("wrong"), "abstain": g("abstained"), "clarification_rate": g("asked"),
            "turns": g("turns"), "cost": g("cost"), "utility": g("utility"),
            "unnecessary_clarification_rate": g("unnecessary_clarification"),
            "unnecessary_given_asked": float(np.mean([x["unnecessary_clarification"] for x in asked])) if asked else 0.0}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--policies", nargs="+", default=["all"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--style", default="mixed", choices=["mixed", "bare", "corrective"])
    ap.add_argument("--tuned", default=str(ROOT / "data/phase5/tuned.json"))
    ap.add_argument("--channel", default="data/phase4/channel.v1.json")
    a = ap.parse_args()
    if "test" in a.splits and not (ROOT / "reports/protocol_freeze.json").exists():
        raise SystemExit("refusing to run on the test split before reports/protocol_freeze.json exists")
    w = World(a.style, a.channel)
    tuned = json.loads(Path(a.tuned).read_text()) if Path(a.tuned).exists() else {}
    pols = make_policies(w.ch, tuned, a.policies)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    variants = w.variants(a.splits)
    summary, pending = {}, 0
    with open(out / "episodes.jsonl", "w") as fh:
        for name, pol in pols.items():
            eps = [run_episode(w, pol, m) for m in variants]
            for e in eps:
                fh.write(json.dumps(e, sort_keys=True, default=float) + "\n")
            pending += sum(e["evaluator"]["status"] == "pending" for e in eps)
            summary[name] = summarise(eps)
    new = w.replies.flush_requests()
    (out / "summary.json").write_text(json.dumps({"splits": a.splits, "style": a.style, "tuned": tuned,
                                                 "pending_episodes": pending, "new_ondemand_requests": new,
                                                 "policies": summary}, indent=2, default=float) + "\n")
    print(f"pending episodes: {pending}  (new on-demand requests: {new})")
    for k, v in summary.items():
        if v:
            print(f"{k:22} n={v['n']:3d} succ={v['success']:.2f} wrong={v['wrong']:.2f} abst={v['abstain']:.2f} "
                  f"ask={v['clarification_rate']:.2f} util={v['utility']:+.3f} unnec={v['unnecessary_clarification_rate']:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
