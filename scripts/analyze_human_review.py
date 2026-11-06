"""Normalise the human review (Excel = authoritative) and compare it with the logged automatic results.

Read-only on every existing artifact. Reads the Excel workbook (via merge_human_review.read_xlsx), the Phase 1
snapshot and data/phase5/dev.v1/episodes.jsonl. Writes only NEW files next to the review package:
  review_human_normalized.csv, human_review_analysis.json
H_* columns are the human's (verbatim in *_raw, standardised in *_norm); A_* columns are ANALYST codings added by
this script from the logged evidence and are labelled as such. Nothing here edits an experimental outcome.

    python scripts/analyze_human_review.py
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from merge_human_review import H_COLS, read_xlsx  # noqa: E402

D = ROOT / "reports/audit/review_package"
QC, RC, AC, FC, OC, NC, RV = H_COLS

# Only standardisation the user authorised (message of 2026-11-02): D19 question rating 'Y' -> 5.
STANDARDISE = {("D19", QC): ("Y", "5", "user confirmed the rating is 5/5, not Y")}

# Analyst codings (NOT human judgments), each tied to evidence checked in the logs / transcripts.
# classes: ASR_INIT, ASR_REPLY, PARSER, POLICY_Q (question choice/wording), POLICY_ASK (asked or did not ask),
#          BELIEF (lexical belief artefact), TIE (designed-ambiguity tie), NONE
CODING = {
    "D01": ("ASR_INIT", "ASR 'Come on.' for 'call mom' at 10 dB; gold not in top-5; all ask EUs < 0 so evpi abstains"),
    "D02": ("POLICY_ASK", "initial max-prob .945; ask_confirm EU .8934 vs answer .8899 (margin .0035)"),
    "D03": ("ASR_INIT+ASR_REPLY+POLICY_Q", "initial ASR 'Cole Dan.'; reply 'Dan' heard as 'Then' (parsed None); generic open question rated 2"),
    "D04": ("ASR_REPLY+PARSER", "reply 'Buy flour' -> ASR 'Bye, flower.' -> parse_closed unparseable; belief unchanged (.9944); success came from the prior"),
    "D05": ("POLICY_ASK", "initial max-prob .872; confirm asked; human: request already clear"),
    "D06": ("NONE", "open question generic (rated 4) but reply parsed and flipped belief from 'record video' to 'record audio only'"),
    "D07": ("TIE", "3-way tie (1/3 each) among calendar/reminder/note; correct answer by hash tie-break; never_ask"),
    "D08": ("ASR_INIT+POLICY_ASK", "ASR 'Place Reelers.' at 5 dB; gold not in top-5; every ask EU < 0 under the model; human would have asked"),
    "D09": ("NONE", "closed intent question; reply 'call' parsed; initial MAP was wrong ('show Priya contact details'), final right"),
    "D10": ("TIE", "3-way tie hotel/flight/car; never_ask answered the tie wrongly"),
    "D11": ("NONE", "direct answer at .986"),
    "D12": ("POLICY_ASK", "always_ask baseline asked a closed city question at max-prob .9973"),
    "D13": ("POLICY_ASK", "confirm asked at .6526; initial MAP already right; necessity debatable per human"),
    "D14": ("NONE", "direct answer at .996 under 5 dB"),
    "D15": ("BELIEF+POLICY_ASK", "'9 p.m.' transcribed correctly but belief split .4998/.4998 between 9 pm and 9 am; confirm resolved it"),
    "D16": ("PARSER+POLICY_Q", "reply 'Seven in the morning' transcribed '7 in the morning.'; gold task scored S_r=-0.658 (grammar mismatch); generic question rated 3"),
    "D17": ("PARSER+POLICY_Q", "reply 'Jazz.' transcribed exactly but ASR word prob .163 -> S_r .326 < 0.5 gate; generic question rated 3"),
    "D18": ("POLICY_ASK", "'cancel it'; max-prob .0389; ask_open EU -.2044 vs abstain 0; human would have asked"),
    "D19": ("NONE", "closed device question; reply parsed; ties resolved; correct"),
    "D20": ("POLICY_Q+TIE", "always_ask asked an INTENT question (book ticket / weather / reminder) though the missing info was the event; post-reply 3-way event tie -> wrong"),
}


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(max(0, c - h), 3), round(min(1, c + h), 3))


def summ(xs):
    xs = sorted(xs)
    if not xs:
        return {}
    q = lambda p: xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))]  # noqa: E731
    return {"n": len(xs), "mean": round(mean(xs), 2), "median": median(xs), "min": xs[0], "max": xs[-1], "q1": q(.25), "q3": q(.75),
            "counts": dict(sorted(Counter(xs).items()))}


def main() -> int:
    out_csv, out_json = D / "review_human_normalized.csv", D / "human_review_analysis.json"
    for p in (out_csv, out_json):
        if p.exists():
            raise SystemExit(f"refusing to overwrite {p}")
    header, xrows, _ = read_xlsx(D / "dialogue_review_20(1).xlsx")
    X = {r["dialogue_id"]: r for r in xrows}
    eps = {}
    for l in (ROOT / "data/phase5/dev.v1/episodes.jsonl").read_text().splitlines():
        e = json.loads(l)
        eps[f"{e['agent']['policy']}|{e['agent']['clip_id']}"] = e
    rows = []
    for i in range(1, 21):
        did = f"D{i:02d}"
        x = X[did]
        e = eps[x["episode_key"]]
        a, v = e["agent"], e["evaluator"]
        r = {"dialogue_id": did, "split": v["split"], "base_id": v["base_id"], "category": v["category"], "snr_db": x["snr_db"],
             "policy": a["policy"], "decision": x["decision"], "final": a["final"], "utility": v["utility"], "auto_flags": x["auto_flags"]}
        norm_notes = []
        for c in (QC, RC, AC, FC, OC, RV):
            raw = x[c]
            nv = raw
            if (did, c) in STANDARDISE:
                old, new, why = STANDARDISE[(did, c)]
                assert raw == old, (did, c, raw)
                nv = new
                norm_notes.append(f"{c}: '{old}' -> '{new}' ({why})")
            r[c + "_raw"], r[c + "_norm"] = raw, nv
        r["H_notes"] = x[NC]
        r["normalization_note"] = "; ".join(norm_notes)
        # automatic evidence
        asked = a["action"] == "ask"
        r["asked"] = asked
        r["question_type"] = a["question"]["type"] if asked else ""
        r["initial_map_correct"] = v["initial_correct"]
        r["final_outcome"] = "success" if v["success"] else ("wrong" if v["wrong"] else "abstain")
        if asked:
            r["reply_parsed"] = a.get("parsed_outcome") is not None
            r["reply_asr_matches_intended"] = (a.get("reply_asr_text", "").strip(" .").lower() == v["simulator_reply"]["text"].strip(" .").lower())
            if a.get("parsed_outcome") is None:
                eff = "reply_unusable"
            elif not v["initial_correct"] and v["success"]:
                eff = "rescued (initial MAP wrong -> final right)"
            elif v["initial_correct"] and v["success"]:
                eff = "confirmed (initial MAP right -> final right)"
            elif v["wrong"]:
                eff = "failed (final wrong)"
            else:
                eff = "abstained after reply"
            if a.get("parsed_outcome") is None and v["success"]:
                eff = "reply_unusable; success from prior"
            if a.get("parsed_outcome") is None and not v["success"]:
                eff = "reply_unusable; " + ("abstained (initial MAP was right)" if v["initial_correct"] else "abstained (initial MAP wrong)")
            r["clarification_effect"] = eff
        else:
            r["reply_parsed"] = r["reply_asr_matches_intended"] = ""
            r["clarification_effect"] = "no clarification"
        cls, ev = CODING[did]
        r["A_failure_classes(analyst)"], r["A_evidence(analyst)"] = cls, ev
        rows.append(r)

    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ------------------------------------------------------------------------------ statistics
    st = {"n": 20}
    asked = [r for r in rows if r["asked"]]
    st["n_asked"], st["n_not_asked"] = len(asked), 20 - len(asked)
    qn = {r["dialogue_id"]: int(r[QC + "_norm"]) for r in asked}
    st["question_rating"] = summ(list(qn.values()))
    for key in ("question_type", "policy"):
        st[f"question_rating_by_{key}"] = {k: summ([qn[r["dialogue_id"]] for r in asked if r[key] == k]) for k in sorted({r[key] for r in asked})}
    st["question_rating_by_final_outcome"] = {k: summ([qn[r["dialogue_id"]] for r in asked if r["final_outcome"] == k]) for k in ("success", "wrong", "abstain")}
    st["reply_appropriateness"] = dict(Counter(r[RC + "_norm"] for r in asked))
    st["audio_asr_consistency"] = dict(Counter(r[AC + "_norm"] for r in rows))
    st["final_action_correct"] = dict(Counter(r[FC + "_norm"] for r in rows))
    st["overall_valid"] = dict(Counter(r[OC + "_norm"] for r in rows))
    # blank consistency
    st["blank_consistency"] = {"not_asked_blank_q_and_r": all(r[QC + "_norm"] == "" and r[RC + "_norm"] == "" for r in rows if not r["asked"]),
                               "asked_filled_q_and_r": all(r[QC + "_norm"] != "" and r[RC + "_norm"] != "" for r in asked)}
    # human vs auto: transcript consistency
    auto_diff = {r["dialogue_id"] for r in rows if {f.strip().split(" ")[0] for f in r["auto_flags"].split(";")} & {"F1", "F8"}}
    human_not_y = {r["dialogue_id"] for r in rows if r[AC + "_norm"] != "Y"}
    st["asr_flag_vs_human"] = {"auto_transcript_differs": sorted(auto_diff), "human_not_Y": sorted(human_not_y), "identical_sets": auto_diff == human_not_y}
    # human vs auto: final action
    ans = [r for r in rows if r["final"] == "answer"]
    agree = [r for r in ans if (r[FC + "_norm"] == "Y") == (r["final_outcome"] == "success")]
    st["final_action_answered"] = {"n": len(ans), "agree": len(agree), "wilson95": wilson(len(agree), len(ans)),
                                   "disagree": [(r["dialogue_id"], r["final_outcome"], r[FC + "_norm"]) for r in ans if r not in agree]}
    ab = [r for r in rows if r["final"] == "abstain"]
    st["final_action_abstained"] = {"n": len(ab), "human": {r["dialogue_id"]: r[FC + "_norm"] for r in ab},
                                    "initial_map_correct": {r["dialogue_id"]: r["initial_map_correct"] for r in ab}}
    # clarification quality vs outcome
    st["clarification_effect"] = dict(Counter(r["clarification_effect"] for r in asked))
    # F10 (unnecessary clarification) vs human text coding
    f10 = sorted(r["dialogue_id"] for r in rows if "F10" in r["auto_flags"])
    st["F10_flagged"] = f10
    # failure class counts (analyst)
    cc = Counter()
    for r in rows:
        for c in r["A_failure_classes(analyst)"].split("+"):
            cc[c] += 1
    st["analyst_failure_class_counts_by_dialogue"] = dict(cc)
    st["snr_vs_category_of_failure"] = {r["dialogue_id"]: (r["snr_db"], r["A_failure_classes(analyst)"]) for r in rows if r["A_failure_classes(analyst)"] != "NONE"}
    st["d01"] = {"audio_asr": X["D01"][AC], "final_correct": X["D01"][FC], "overall": X["D01"][OC], "note": X["D01"][NC]}
    out_json.write_text(json.dumps(st, indent=1, default=str) + "\n")
    print(json.dumps(st, indent=1, default=str))
    for r in rows:
        print(r["dialogue_id"], r["policy"], r["decision"], "->", r["final_outcome"], "| init_MAP_ok", r["initial_map_correct"], "|", r["clarification_effect"], "| parsed", r["reply_parsed"], "| Q", r[QC + "_norm"], "R", r[RC + "_norm"], "A", r[AC + "_norm"], "F", r[FC + "_norm"], "O", r[OC + "_norm"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
