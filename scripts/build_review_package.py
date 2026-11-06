"""Build the manual-review package and audit statistics from EXISTING dev episodes (read-only).

Reads data/phase5/dev.v1/episodes.jsonl (train + val only; the held-out test split is never opened),
the Phase 1 snapshot, the audio manifests and ASR outputs. Re-runs nothing: no ASR, TTS, policy or
tuning code is executed. Writes only under reports/audit/.

    python scripts/build_review_package.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.audio_service import reply_clip_id  # noqa: E402
from src.ontology import tokens  # noqa: E402

SNAP = ROOT / "data/phase1/snapshots/phase1.v2.0+05471eebec39"
OUT = ROOT / "reports/audit"
REVIEW = OUT / "review_package"
SEED = 1234
N_PICK = 20
POLICIES_SHOWN = ["nm_voi", "noise_indep_voi", "evpi", "always_ask", "confidence_threshold", "never_ask"]


def jl(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def h(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def norm(s: str) -> list:
    return tokens(s or "")


# ------------------------------------------------------------------------------ data ---------
eps = jl(ROOT / "data/phase5/dev.v1/episodes.jsonl")
assert all(e["evaluator"]["split"] in ("train", "val") for e in eps), "dev episodes must be train/val only"
recs = {r["base_id"]: r for r in jl(SNAP / "base_items.jsonl")}
man = jl(ROOT / "data/phase2/audio.v1/audio_manifest.jsonl")
man_od = jl(ROOT / "data/phase2/ondemand.v1/ondemand_manifest.jsonl")
by_clip = {m["clip_id"]: m for m in man}
by_clip.update({m["clip_id"]: m for m in man_od})
asr = {r["clip_id"]: r for r in jl(ROOT / "data/phase3/asr.v1/asr_results.jsonl")}
asr.update({r["clip_id"]: r for r in jl(ROOT / "data/phase3/asr_ondemand.v1/asr_results.jsonl")})
E = {(e["agent"]["policy"], e["agent"]["clip_id"]): e for e in eps}
clip_ids = sorted({e["agent"]["clip_id"] for e in eps})


def decision_label(e) -> str:
    a = e["agent"]
    return f"ask_{a['question']['type']}" if a["action"] == "ask" else a["action"]


def intended_outcome(e):
    """Outcome index the simulator's reply was meant to produce (evaluator-side view)."""
    q, rt = e["agent"].get("question"), e["evaluator"].get("simulator_reply", {}).get("type")
    if not q or not rt:
        return None
    if q["type"] == "confirm":
        return {"yes": 0, "no_bare": 1, "no_correction": 1}.get(rt)
    if q["type"] == "closed":
        if rt == "none_of_these":
            return len(q["options"])
        if rt == "select":
            return [o.lower() for o in q["options"]].index(e["evaluator"]["simulator_reply"]["text"].rstrip(".").lower())
        return None
    return "n/a"


COMPAT = {"open": {"answer", "off_target"}, "closed": {"select", "none_of_these", "off_target"},
          "confirm": {"yes", "no_bare", "no_correction"}}


def flags(e) -> list:
    out, a, v = [], e["agent"], e["evaluator"]
    rec = recs[v["base_id"]]["private"]
    top = a["initial_top"]
    if len(top) > 1 and abs(top[0][1] - top[1][1]) < 1e-3:
        out.append("F6 initial near-tie between two top tasks (winner chosen by hash jitter)")
    if v["snr_nominal_db"] is not None and a["snr_bucket_est"] != str(v["snr_nominal_db"]):
        out.append(f"F7 blind SNR bucket {a['snr_bucket_est']} != nominal {v['snr_nominal_db']}")
    if v["snr_nominal_db"] is None and a["snr_bucket_est"] != "clean":
        out.append(f"F7 blind SNR bucket {a['snr_bucket_est']} != clean")
    if norm(a["asr_text"]) != norm(rec["reference_text"]):
        out.append("F8 initial ASR transcript differs from reference text")
    if all(r[0] != v["gold"] for r in top):
        out.append("F13 gold task not in agent's initial top-5")
    if a["action"] == "ask":
        sr = v.get("simulator_reply", {})
        if sr.get("type") not in COMPAT[a["question"]["type"]]:
            out.append(f"F5 reply type {sr.get('type')} incompatible with {a['question']['type']} question")
        if sr.get("type") == "off_target":
            out.append("F4 simulator reply off_target (question asked about a variable the hidden task lacks / nothing unresolved)")
        if norm(a.get("reply_asr_text", "")) != norm(sr.get("text", "")):
            out.append("F1 reply ASR transcript differs from intended reply text")
        po, io = a.get("parsed_outcome"), intended_outcome(e)
        if po is None:
            out.append("F2 reply not usable by the agent (parsed outcome None; belief not updated)")
        elif io not in (None, "n/a") and po != io:
            out.append(f"F3 parsed outcome {po} != intended outcome {io}")
        if v["success"] is False and v["wrong"] and (po is not None) and (io in ("n/a", po)):
            out.append("F9 final answer wrong although the reply was parsed as intended")
        if v["unnecessary_clarification"]:
            out.append("F10 clarification asked although initial MAP task was already acceptable")
    return out


# --------------------------------------------------------------------------- selection -------
cand = []
for cid in clip_ids:
    for p in POLICIES_SHOWN:
        e = E.get((p, cid))
        if e and e["evaluator"]["status"] == "complete":
            cand.append(e)

disagree = []
for cid in clip_ids:
    a, b = E[("nm_voi", cid)], E[("noise_indep_voi", cid)]
    if decision_label(a) != decision_label(b) or a["agent"]["final"] != b["agent"]["final"]:
        disagree.append(cid)

picked, used_base = [], set()


def tags(e):
    v = e["evaluator"]
    return [("cat", v["category"]), ("snr", str(v["snr_nominal_db"])), ("dec", decision_label(e)),
            ("pol", e["agent"]["policy"]), ("fin", e["agent"]["final"])]


def tag_counts():
    c = Counter()
    for e in picked:
        c.update(tags(e))
    return c


def take(e):
    picked.append(e)
    used_base.add(e["evaluator"]["base_id"])


def order_key(e):
    return h(f"{SEED}|{e['agent']['policy']}|{e['agent']['clip_id']}")


# seed 1: nm_voi vs noise_indep_voi disagreements (central hypothesis) -- up to 3, show nm_voi side
for cid in sorted(disagree, key=lambda c: h(f"{SEED}|dis|{c}"))[:3]:
    take(E[("nm_voi", cid)])
# seed 2: flagged inconsistencies in asking dialogues (F2/F3/F4/F9) -- up to 3
bad = [e for e in sorted(cand, key=order_key)
       if e["agent"]["action"] == "ask" and e["evaluator"]["base_id"] not in used_base
       and any(f[:2] in ("F2", "F3", "F4", "F9") for f in flags(e))]
for e in bad[:3]:
    if e["evaluator"]["base_id"] not in used_base:
        take(e)
# greedy fill for coverage of category, SNR, decision type, policy
while len(picked) < N_PICK:
    c = tag_counts()
    pool = [e for e in cand if e["evaluator"]["base_id"] not in used_base]
    best = max(pool, key=lambda e: (sum(1.0 / (1 + c[t]) for t in tags(e)), order_key(e)))
    take(best)
picked.sort(key=lambda e: (e["evaluator"]["category"], str(e["evaluator"]["snr_nominal_db"]), e["agent"]["policy"]))

# ------------------------------------------------------------------------- package ----------
if REVIEW.exists():
    shutil.rmtree(REVIEW)
(REVIEW / "audio").mkdir(parents=True)


def copy_audio(clip_id: str, label: str):
    m = by_clip.get(clip_id)
    if not m:
        return None, None, "clip missing from manifests"
    src = ROOT / m["path"]
    if not src.exists():
        return None, m, "audio file missing"
    dst = REVIEW / "audio" / f"{label}__{clip_id}.wav"
    shutil.copyfile(src, dst)
    ok = hashlib.sha256(src.read_bytes()).hexdigest() == m["sha256"]
    return dst.relative_to(REVIEW), m, "" if ok else "sha256 mismatch vs manifest"


rows, md = [], []
md.append("# Manual review package: 20 closed-loop dialogues\n")
md.append("**Nothing here is human-reviewed.** All fields in `review_checklist.csv` marked *(blank)* are for you to fill in. "
          "Data are SYNTHETIC (hand-authored text, Piper TTS, DEMAND cafeteria noise). Dialogues are drawn only from the "
          "train/val dev episodes (`data/phase5/dev.v1`); the held-out test split was not opened. Audio files are byte-for-byte "
          "copies (sha256 checked against the Phase 2 manifests); originals are untouched.\n")
md.append("**Listening in VS Code:** open this file with *Markdown: Open Preview* (Ctrl+Shift+V); the `<audio>` players "
          "work in the preview. If a player does not render, click the `.wav` link (VS Code's built-in media preview plays WAV), "
          "or open `index.html` in a browser/Live Preview.\n")
md.append("Legend of automatic flags (informational, not verdicts): F1 reply ASR differs from intended text; F2 reply unusable "
          "(belief not updated); F3 parsed outcome != intended; F4 off-target simulator reply; F5 reply type incompatible with "
          "question; F6 near-tie initial belief; F7 blind SNR bucket != nominal; F8 initial ASR != reference; F9 wrong answer "
          "despite correctly parsed reply; F10 unnecessary clarification; F13 gold not in agent's initial top-5.\n")
html = ["<!doctype html><meta charset=utf-8><title>NM-VoI review</title><body style='font-family:sans-serif;max-width:900px;margin:auto'>",
        "<h1>NM-VoI manual review (not human-reviewed yet)</h1>"]

for i, e in enumerate(picked, 1):
    a, v = e["agent"], e["evaluator"]
    base = recs[v["base_id"]]["private"]
    did = f"D{i:02d}"
    fl = flags(e)
    init_audio, init_m, w1 = copy_audio(a["clip_id"], f"{did}_initial")
    rep_audio = rep_m = None
    w2 = ""
    reply_clip = None
    if a["action"] == "ask":
        reply_clip, _ = reply_clip_id(SEED, v["base_id"], v["simulator_reply"]["text"], v["snr_nominal_db"])
        rep_audio, rep_m, w2 = copy_audio(reply_clip, f"{did}_reply")
    for w in (w1, w2):
        if w:
            fl.append(f"F12 {w}")
    q = a.get("question")
    gold_acc = base["acceptable_tasks"]
    rows.append({
        "dialogue_id": did, "episode_key": f"{a['policy']}|{a['clip_id']}", "split": v["split"], "base_id": v["base_id"],
        "category": v["category"], "snr_db": "clean" if v["snr_nominal_db"] is None else v["snr_nominal_db"],
        "policy": a["policy"], "decision": decision_label(e), "final": a["final"],
        "utility": v["utility"], "auto_flags": "; ".join(f.split(" ")[0] for f in fl),
        "initial_audio": str(init_audio or ""), "reply_audio": str(rep_audio or ""),
        "H_question_relevance(1-5 or Y/N)": "", "H_reply_appropriateness": "", "H_audio_asr_consistency": "",
        "H_final_action_correct": "", "H_overall_valid(Y/N/unsure)": "", "H_notes": "", "H_reviewed_by_human": "NO",
    })
    md.append(f"\n---\n\n## {did} · `{v['base_id']}` · {v['category']} · {rows[-1]['snr_db']} dB · {a['policy']} · split={v['split']}\n")
    md.append(f"- **Episode key:** `{a['policy']}|{a['clip_id']}` (in `data/phase5/dev.v1/episodes.jsonl`)")
    md.append(f"- **Original utterance (reference text):** “{base['reference_text']}”")
    md.append(f"- **Intended meaning (hidden gold):** {v['gold']}  ·  acceptable set: `{json.dumps(gold_acc)}`")
    md.append(f"- **Initial audio:** [{init_audio}]({init_audio})  \n  <audio controls src=\"{init_audio}\"></audio>")
    md.append(f"- **ASR of original (agent saw):** “{a['asr_text']}”  ·  blind SNR bucket est.: {a['snr_bucket_est']}")
    md.append(f"- **Initial belief top-5:** " + "; ".join(f"{t} ({p:.3f})" for t, p in a["initial_top"]) +
              f"  ·  max-prob {a['initial_max_prob']}, entropy {a['initial_entropy']}")
    if a["eus"]:
        md.append(f"- **Expected utilities considered:** `{json.dumps(a['eus'])}`")
    md.append(f"- **Initial decision:** **{decision_label(e)}**")
    if q:
        sr = v["simulator_reply"]
        md.append(f"- **Clarification question ({q['type']}):** “{q['text']}”"
                  + (f"  (variable: {q['variable']}; options: {q['options']})" if q["type"] == "closed" else ""))
        rel = {"open": "generic open question: simulator restates the unresolved information",
               "closed": "simulator answers with the offered option equal to its hidden value, else 'None of those.'",
               "confirm": "simulator says Yes only if the proposal is acceptable, else No (bare or corrective)"}[q["type"]]
        md.append(f"- **Intended simulated reply:** “{sr['text']}” (type `{sr['type']}`)  ·  relevance rule: {rel}")
        md.append(f"- **Reply audio:** [{rep_audio}]({rep_audio})  \n  <audio controls src=\"{rep_audio}\"></audio>")
        md.append(f"- **ASR of reply (agent saw):** “{a.get('reply_asr_text')}”  ·  parsed outcome: {a.get('parsed_outcome')}")
        md.append("- **Updated belief top-3:** " + "; ".join(f"{t} ({p:.3f})" for t, p in a.get("updated_top", [])))
    md.append(f"- **Final decision:** **{a['final']}**" + (f" → “{a['final_task']}”" if a["final_task"] else "")
              + f"  ·  success={v['success']}, wrong={v['wrong']}, reward={v['reward']:+.1f}, cost={v['cost']:.2f}, "
                f"**utility={v['utility']:+.2f}**")
    md.append("- **Automatic flags:** " + ("; ".join(fl) if fl else "none"))
    md.append("- **Human assessment:** *(blank, see `review_checklist.csv`, row " + did + ")*")
    html.append(f"<h2>{did} {v['base_id']} {v['category']} {rows[-1]['snr_db']} dB {a['policy']}</h2>"
                f"<p>Reference: {base['reference_text']}<br>ASR: {a['asr_text']}<br>Decision: {decision_label(e)} → {a['final']} "
                f"(utility {v['utility']:+.2f})</p><audio controls src='{init_audio}'></audio>"
                + (f"<p>Q: {q['text']}<br>Reply: {v['simulator_reply']['text']} → ASR: {a.get('reply_asr_text')}</p>"
                   f"<audio controls src='{rep_audio}'></audio>" if q else ""))

(REVIEW / "dialogue_review.md").write_text("\n".join(md) + "\n")
(REVIEW / "index.html").write_text("\n".join(html) + "\n")
with open(REVIEW / "review_checklist.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
(REVIEW / "selection.json").write_text(json.dumps({
    "rule": "3 nm_voi-vs-noise_indep_voi disagreements, up to 3 flagged asking dialogues (F2/F3/F4/F9), rest greedy coverage of "
            "category/SNR/decision/policy; one dialogue per base item; deterministic (seed 1234 hash order); dev (train/val) only",
    "episodes": [r["episode_key"] for r in rows]}, indent=1) + "\n")

# ------------------------------------------------------------------ whole-dev audit stats -----
st = {}
nm = {c: E[("nm_voi", c)] for c in clip_ids}
ni = {c: E[("noise_indep_voi", c)] for c in clip_ids}
st["n_dev_clips"] = len(clip_ids)
st["nm_vs_noise_indep_initial_action_differs"] = sum(decision_label(nm[c]) != decision_label(ni[c]) for c in clip_ids)
st["nm_vs_noise_indep_final_outcome_differs"] = sum(
    (nm[c]["agent"]["final"], nm[c]["agent"]["final_task"]) != (ni[c]["agent"]["final"], ni[c]["agent"]["final_task"]) for c in clip_ids)
st["nm_vs_noise_indep_utility_differs"] = sum(abs(nm[c]["evaluator"]["utility"] - ni[c]["evaluator"]["utility"]) > 1e-9 for c in clip_ids)
st["differ_by_snr"] = dict(Counter(str(nm[c]["evaluator"]["snr_nominal_db"]) for c in clip_ids
                                   if decision_label(nm[c]) != decision_label(ni[c])))
st["utility_by_split"] = {}
for sp in ("train", "val"):
    cs = [c for c in clip_ids if nm[c]["evaluator"]["split"] == sp]
    st["utility_by_split"][sp] = {"n_clips": len(cs), "n_items": len({nm[c]["evaluator"]["base_id"] for c in cs})}
    for p in POLICIES_SHOWN:
        st["utility_by_split"][sp][p] = round(sum(E[(p, c)]["evaluator"]["utility"] for c in cs) / len(cs), 4)
allf = Counter()
asked = Counter()
for e in eps:
    if e["agent"]["policy"] == "nm_voi":
        asked["nm_voi_episodes"] += 1
        for f in flags(e):
            allf[f.split(" ")[0]] += 1
        if e["agent"]["action"] == "ask":
            asked["nm_voi_asked"] += 1
            asked["reply_used"] += e["agent"].get("parsed_outcome") is not None
            asked["reply_asr_exact"] += norm(e["agent"].get("reply_asr_text", "")) == norm(e["evaluator"]["simulator_reply"]["text"])
st["nm_voi_flag_counts"] = dict(allf)
st["nm_voi_asked"] = dict(asked)
near = [e for e in eps if e["agent"]["policy"] == "never_ask" and e["evaluator"]["category"] == "none"]
st["none_category_never_ask_success"] = round(sum(e["evaluator"]["success"] for e in near) / len(near), 3)
st["none_category_near_tie_rate"] = round(sum(any(f.startswith("F6") for f in flags(e)) for e in near) / len(near), 3)
conf = Counter((str(e["evaluator"]["snr_nominal_db"]), e["agent"]["snr_bucket_est"]) for e in eps if e["agent"]["policy"] == "nm_voi")
st["snr_bucket_confusion_nominal_vs_est"] = {f"{k[0]}->{k[1]}": n for k, n in sorted(conf.items())}
rq = Counter()
for e in eps:
    if e["agent"]["policy"] == "nm_voi" and e["agent"]["action"] == "ask":
        rq[(e["agent"]["question"]["type"], e["evaluator"]["simulator_reply"]["type"])] += 1
st["nm_voi_question_type_x_reply_type"] = {f"{k[0]}/{k[1]}": n for k, n in sorted(rq.items())}
# provenance of reply audio
src = Counter()
for e in eps:
    if e["agent"]["policy"] == "nm_voi" and e["agent"]["action"] == "ask":
        cid, _ = reply_clip_id(SEED, e["evaluator"]["base_id"], e["evaluator"]["simulator_reply"]["text"], e["evaluator"]["snr_nominal_db"])
        src["phase2_inventory" if cid in {m["clip_id"] for m in man} else "ondemand"] += 1
        src["has_audio_file"] += (ROOT / by_clip[cid]["path"]).exists() if cid in by_clip else 0
        src["has_asr_record"] += cid in asr
st["nm_voi_reply_provenance"] = dict(src)
# channel partition check: every train-count clip noise partition
chj = json.loads((ROOT / "data/phase4/channel.v1.json").read_text())
st["channel_fit_on"] = chj["estimate"]["fit_on"]
st["channel_train_total_trials"] = sum(v[1] for q in chj["estimate"]["train_counts"].values() for v in q.values())
st["train_reply_clip_noise_partitions"] = dict(Counter(m["noise"]["partition"] for m in man if m["kind"] == "reply" and m["split"] == "train" and m.get("noise")))
st["val_reply_clip_noise_partitions"] = dict(Counter(m["noise"]["partition"] for m in man if m["kind"] == "reply" and m["split"] == "val" and m.get("noise")))
(OUT / "audit_stats.json").write_text(json.dumps(st, indent=1, default=str) + "\n")
print(json.dumps(st, indent=1, default=str))
print("picked:", [(r["dialogue_id"], r["category"], r["snr_db"], r["policy"], r["decision"], r["auto_flags"]) for r in rows])
