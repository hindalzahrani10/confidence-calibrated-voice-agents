"""Validate the Phase 1 feasibility dataset (schema phase1.v2).

Written independently of scripts/author_phase1.py: it imports nothing from it,
holds its own copy of the approved allocation and slot rules, and re-derives
every recovery status, exclusion and leakage property from the raw
annotations. Passing these checks shows the file is internally consistent. It
does NOT show the annotations are correct or the dataset scientifically valid.

    python3 scripts/validate_phase1.py                 # default paths
    python3 scripts/validate_phase1.py --src src       # also scan agent code for private fields

Exits non-zero on any failure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent

APPROVED = {"none": (8, 3, 2), "acoustic": (8, 2, 3), "intent": (7, 3, 2), "underspecified": (7, 2, 3)}
SPLITS = ("train", "val", "test")
AXIS_FOR = {"none": "none", "acoustic": "acoustic", "intent": "intent", "underspecified": "slot"}
SCHEMA_VERSION = "phase1.v2"
TOP_KEYS = {"schema_version", "base_id", "split", "source", "agent_visible", "private"}
PRIVATE_KEYS = {
    "ambiguity_type", "distinguishing_axis", "axis_rationale", "gold_prior", "acoustic_confusable", "conditions",
    "reference_text", "gold_task", "utterance_slots", "required_slots", "at_risk_slots", "say", "intent_families",
    "alternatives", "candidate_set_exhaustive", "acceptable_tasks", "acceptable_note", "probes",
    "not_applicable_questions", "scoring", "annotation", "meta",
}
CONFUSABLE_KINDS = {"minimal_pair", "homophone", "name_collision", "near_homophone"}
REPLY_TYPES = {"answer", "select", "none_of_these", "yes", "no_bare", "no_correction"}
VARIANT_ID = re.compile(r"^u[0-9a-f]{8}-[a-d]$")
# Words that would tell an agent the category, condition or task if they leaked into an agent-visible id.
LEAK_WORDS = ("none", "acoustic", "intent", "underspec", "clean", "snr", "train", "val", "test", "p1_")
# Private field names that agent-side code must not reference (static scan).
PRIVATE_TOKENS = ("gold_task", "acceptable_tasks", "utterance_slots", "at_risk_slots", "alternatives",
                  "not_applicable_questions", "proposal_is_correct", "candidate_set_exhaustive")
LEGACY_TOKENS = ("gold_answers", "clarification_reply", "reference_text", "ambiguity_type")

INT_RANGES = {"temperature_c": (10, 30), "duration_min": (1, 180), "level": (0, 100), "value": (0, 100),
              "party_size": (1, 12), "quantity": (1, 10), "lead_time_min": (1, 1440)}
ENUMS = {"direction": {"up", "down"}}


class Report:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def check(self, ok: bool, msg: str) -> bool:
        if not ok:
            self.errors.append(msg)
        return bool(ok)


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s'-]", " ", str(s).lower())).strip()


def has_phrase(text: str, phrase: str) -> bool:
    return f" {norm(phrase)} " in f" {norm(text)} "


def valid_slot(key: str, value) -> bool:
    if key in INT_RANGES:
        lo, hi = INT_RANGES[key]
        return isinstance(value, int) and not isinstance(value, bool) and lo <= value <= hi
    if key in ENUMS:
        return value in ENUMS[key]
    if key == "time":
        m = re.fullmatch(r"([01]\d|2[0-3]):([0-5]\d)", str(value))
        return bool(m)
    return isinstance(value, str) and bool(value.strip())


def matches(task: dict, pattern: dict) -> bool:
    if task.get("intent") != pattern.get("intent"):
        return False
    ps, ts = pattern.get("slots", {}), task.get("slots", {})
    return set(ps) == set(ts) and all(pv == "*" or ts[k] == pv for k, pv in ps.items())


def derive(required, known, explicit, assumed=None):
    got = set(known) | set(explicit) | set(assumed or {})
    unresolved = [k for k in required if k not in got]
    helped = any(k in required and k not in known for k in list(explicit) + list(assumed or {}))
    status = "full" if not unresolved else ("partial" if helped else "none")
    return status, unresolved


def check_probe(bid: str, p: dict, P: dict, rep: Report) -> None:
    """All per-probe semantic checks; `P` is the record's private block."""
    pid = p.get("probe_id", f"{bid}.?")
    gt = P["gold_task"]
    gold = {**gt["slots"], "intent": gt["intent"]}
    required, known_cfg = P["required_slots"], P["utterance_slots"]
    at_risk = P["at_risk_slots"]
    known = {k: gold[k] for k in gold if (k in known_cfg and k not in at_risk) or (k == "intent" and "intent" not in required)}
    say = P["say"]
    qt, rp = p.get("question_type"), p.get("reply", {})
    rtype, rtext = rp.get("type"), rp.get("text", "")
    sup, conf = rp.get("supplies", {}), rp.get("confirms", {})
    rec = p.get("recovery", {})
    rep.check(qt in {"open", "closed", "confirm"}, f"{pid}: bad question_type {qt!r}")
    rep.check(rtype in REPLY_TYPES, f"{pid}: bad reply type {rtype!r}")
    rep.check(bool(str(p.get("question", "")).strip()) and bool(rtext.strip()), f"{pid}: empty question or reply text")
    rep.check(rp.get("entailment") == "literal", f"{pid}: Phase 1 replies must be entailment 'literal'")
    for k, v in {**sup, **conf}.items():
        rep.check(gold.get(k) == v, f"{pid}: reply value {k}={v!r} != gold {gold.get(k)!r}")

    exclusions = rec.get("exclusions", [])
    contradicted: list[str] = []
    covers = p.get("covers", [])
    rep.check(set(covers) <= set(gold), f"{pid}: covers {covers} outside gold keys")

    if qt == "open":
        rep.check(rtype == "answer", f"{pid}: open question needs reply type 'answer'")
        rep.check(set(sup) == set(covers) and not conf, f"{pid}: open reply must supply exactly the covered slots")
        rep.check(not exclusions, f"{pid}: open reply must carry no exclusions")
        for k in sup:
            rep.check(k in say and has_phrase(rtext, say[k]),
                      f"{pid}: reply text lacks the literal phrase for {k!r}")
        for k in set(required) - set(known):  # unresolved or at-risk: the question must not state them
            if k in say:
                rep.check(not has_phrase(p["question"], say[k]), f"{pid}: question reveals {k!r}")
    elif qt == "closed":
        opts, vals = p.get("options", []), p.get("option_values", [])
        rep.check(len(opts) >= 2 and len(opts) == len(vals), f"{pid}: closed question needs >=2 options with values")
        rep.check(len(set(map(str, vals))) == len(vals), f"{pid}: duplicate option values")
        rep.check(len(covers) == 1, f"{pid}: closed question must cover exactly one key")
        target = covers[0] if covers else None
        has_gold = target in gold and gold[target] in vals
        rep.check(p.get("options_contain_gold") is has_gold, f"{pid}: options_contain_gold flag is wrong")
        oo = p.get("option_order", {})
        rep.check(bool(oo.get("seed")) and bool(oo.get("method")), f"{pid}: option order seed/method not recorded")
        for o in opts:
            rep.check(has_phrase(p["question"], o), f"{pid}: option {o!r} missing from question text")
        if has_gold:
            idx = vals.index(gold[target])
            rep.check(oo.get("gold_position") == idx, f"{pid}: recorded gold_position != actual {idx}")
            rep.check(rtype == "select" and sup == {target: gold[target]} and not conf and not exclusions,
                      f"{pid}: gold-containing closed question needs a 'select' reply for the gold option")
            rep.check(norm(rtext) == norm(opts[idx]), f"{pid}: selected reply must equal the gold option text")
        else:
            rep.check(rtype == "none_of_these", f"{pid}: gold-omitting question needs 'none_of_these'")
            rep.check(not sup and not conf, f"{pid}: 'none_of_these' must not supply or confirm any value")
            rep.check(oo.get("gold_position") is None, f"{pid}: gold_position must be null when gold is omitted")
            rep.check(exclusions == [{"kind": "rejected_options", "target": target, "values": vals}],
                      f"{pid}: exclusion evidence must list exactly the rejected options")
            for o in opts:
                rep.check(target not in say or not has_phrase(o, say[target]),
                          f"{pid}: gold phrase appears inside a supposedly gold-free option {o!r}")
    elif qt == "confirm":
        prop = p.get("proposed_task", {})
        correct = prop == {"intent": gt["intent"], "slots": gt["slots"]}
        rep.check(p.get("proposal_is_correct") is correct, f"{pid}: proposal_is_correct flag is wrong")
        asserted = {**prop.get("slots", {}), "intent": prop.get("intent")}
        if correct:
            rep.check(rtype == "yes", f"{pid}: a correct proposal must receive 'yes'")
            rep.check(conf == asserted and not sup and not exclusions, f"{pid}: 'yes' confirms the proposal and nothing else")
        else:
            rep.check(rtype in {"no_bare", "no_correction"}, f"{pid}: an incorrect proposal must receive a 'no'")
            rep.check(any(prop == a["task"] for a in P["alternatives"]), f"{pid}: incorrect proposal is not a listed plausible alternative")
            rep.check(not conf, f"{pid}: a 'no' confirms nothing")
            rep.check(exclusions == [{"kind": "rejected_proposal", "task": prop, "joint": True}],
                      f"{pid}: the rejected proposal must be recorded as joint exclusion evidence")
            rep.check(norm(rtext).startswith("no"), f"{pid}: negative reply must begin with 'no'")
            diff = [k for k in asserted if gold.get(k) != asserted[k]]
            rep.check(bool(diff), f"{pid}: incorrect proposal does not differ from gold")
            contradicted = diff
            if rtype == "no_bare":
                rep.check(not sup, f"{pid}: bare 'no' must not supply any value")
            elif rtype == "no_correction":
                rep.check(set(sup) == set(diff) and bool(sup), f"{pid}: correction must supply exactly the wrong keys")
                for k in sup:
                    rep.check(k in say and has_phrase(rtext, say[k]), f"{pid}: correction text lacks phrase for {k!r}")
            if rtype == "no_bare":
                rep.check(rtext.strip().lower().rstrip(".!") == "no", f"{pid}: bare no reply text should be 'No.'")
        rep.check(sorted(covers) == sorted(required), f"{pid}: a confirmation covers the full required task")

    # ----- recovery: re-derive and compare --------------------------------------------------
    explicit = {**sup, **conf}
    known_p = {k: v for k, v in known.items() if k not in contradicted}
    eff = list(required) + [k for k in contradicted if k not in required]
    rep.check(rec.get("from_utterance") == known, f"{pid}: from_utterance != derived {known}")
    rep.check(rec.get("from_reply") == explicit, f"{pid}: from_reply != supplies+confirms")
    rep.check(sorted(rec.get("contradicted_utterance_support", [])) == sorted(k for k in contradicted if k in known),
              f"{pid}: contradicted_utterance_support wrong")
    status, unresolved = derive(eff, known_p, explicit)
    st = rec.get("strict", {})
    rep.check(st.get("status") == status, f"{pid}: strict status {st.get('status')!r} != derived {status!r}")
    rep.check(sorted(st.get("unresolved_slots", [])) == sorted(unresolved),
              f"{pid}: strict unresolved {st.get('unresolved_slots')} != derived {unresolved}")
    if rtype in {"no_bare", "none_of_these"}:
        rep.check(not explicit, f"{pid}: rejection replies carry no recovered values")
        rep.check(status != "full" and st.get("status") != "full", f"{pid}: a bare rejection can never be full recovery")
    assumed = {}
    if qt == "confirm" and rtype == "no_correction":
        wa = {**p["proposed_task"]["slots"], "intent": p["proposed_task"]["intent"]}
        assumed = {k: v for k, v in wa.items() if k in eff and k not in explicit and gold[k] == v}
    lstatus, lunres = derive(eff, known_p, explicit, assumed)
    ln = rec.get("lenient", {})
    rep.check(ln.get("status") == lstatus and sorted(ln.get("unresolved_slots", [])) == sorted(lunres),
              f"{pid}: lenient annotation != derived ({lstatus}, {lunres})")
    rep.check(ln.get("assumed_accepted", {}) == assumed, f"{pid}: lenient assumed_accepted wrong")
    if not assumed:
        rep.check(ln.get("status") == st.get("status"), f"{pid}: lenient may only differ via assumed acceptances")
    # exclusion evidence must never be a recovered value
    for e in exclusions:
        if e.get("kind") == "rejected_options":
            rep.check(gold[e["target"]] not in e["values"], f"{pid}: gold value listed as excluded")
            rep.check(e["target"] not in explicit, f"{pid}: excluded target also shown as recovered")


def check_record(r: dict, snr_levels: list, rep: Report) -> None:
    bid = r.get("base_id", "<missing>")
    rep.check(set(r) == TOP_KEYS, f"{bid}: top-level keys {sorted(set(r) ^ TOP_KEYS)} differ from schema")
    if set(r) != TOP_KEYS:
        return
    rep.check(r["schema_version"] == SCHEMA_VERSION, f"{bid}: schema_version")
    rep.check(r["source"] == "authored_synthetic", f"{bid}: provenance source must be authored_synthetic")
    rep.check(r["split"] in SPLITS, f"{bid}: bad split {r['split']!r}")
    P, V = r["private"], r["agent_visible"]
    rep.check(set(P) == PRIVATE_KEYS, f"{bid}: private keys differ: {sorted(set(P) ^ PRIVATE_KEYS)}")
    if set(P) != PRIVATE_KEYS:
        return
    rep.check(P["meta"].get("external_dataset") is None and P["meta"].get("provenance") == "authored_synthetic",
              f"{bid}: provenance must be authored_synthetic with no external dataset")
    ann = P["annotation"]
    rep.check(ann.get("status") == "ai_drafted_unreviewed" and ann.get("agreement") is None
              and ann.get("annotator_ids") == ["claude_draft"], f"{bid}: review status must be explicit and unfabricated")

    # ----- agent-visible block (leakage) ----------------------------------------------------
    rep.check(set(V) == {"audio_variants"}, f"{bid}: agent_visible must contain only audio_variants")
    vs = V.get("audio_variants", [])
    rep.check(len(vs) == len(snr_levels), f"{bid}: need {len(snr_levels)} variants, found {len(vs)}")
    text_blob = json.dumps(V).lower()
    for v in vs:
        rep.check(set(v) == {"variant_id", "audio_path", "audio_status"}, f"{bid}: visible variant keys")
        rep.check(bool(VARIANT_ID.match(str(v.get("variant_id", "")))), f"{bid}: variant_id {v.get('variant_id')!r} not opaque")
        rep.check(v.get("audio_path") is None and v.get("audio_status") == "planned", f"{bid}: audio must be null/planned")
    id_blob = " ".join(str(v.get("variant_id", "")).lower() for v in vs)
    for w in LEAK_WORDS:
        rep.check(w not in id_blob, f"{bid}: an agent-visible id contains leak word {w!r}")
    gt = P["gold_task"]
    leak_terms = [gt["intent"], *[str(v) for v in gt["slots"].values() if not str(v).isdigit()],
                  P["reference_text"], bid, bid[3:]]
    for t in leak_terms:
        rep.check(len(str(t)) < 3 or str(t).lower() not in text_blob, f"{bid}: agent_visible leaks {t!r}")

    # ----- conditions ------------------------------------------------------------------------
    cond = P["conditions"]
    rep.check([c.get("snr_db") for c in cond] == snr_levels, f"{bid}: planned SNRs {[c.get('snr_db') for c in cond]} != {snr_levels}")
    rep.check(sorted(c["variant_id"] for c in cond) == sorted(v["variant_id"] for v in vs), f"{bid}: conditions/variants id mismatch")

    # ----- category, axis, slots -------------------------------------------------------------
    cat = P["ambiguity_type"]
    rep.check(cat in APPROVED, f"{bid}: unknown ambiguity_type")
    if cat not in APPROVED:
        return
    rep.check(P["distinguishing_axis"] == AXIS_FOR[cat], f"{bid}: axis {P['distinguishing_axis']!r} inconsistent with category {cat!r}")
    gold = {**gt["slots"], "intent": gt["intent"]}
    rep.check(bool(gt.get("domain")) and bool(gt["intent"]), f"{bid}: gold_task incomplete")
    for k, v in gt["slots"].items():
        rep.check(valid_slot(k, v), f"{bid}: implausible/invalid gold slot {k}={v!r}")
    cf = P["acoustic_confusable"]
    rep.check((cf is not None) == (cat == "acoustic"), f"{bid}: acoustic_confusable set iff acoustic")
    intent_unresolved = cat == "intent" or (cf is not None and cf.get("affects") == "intent")
    required = list(gt["slots"]) + (["intent"] if intent_unresolved else [])
    rep.check(sorted(P["required_slots"]) == sorted(required), f"{bid}: required_slots {P['required_slots']} != derived {required}")
    required = list(P["required_slots"])  # order is irrelevant; membership was just verified
    at_risk = [cf["affects"]] if cf else []
    rep.check(P["at_risk_slots"] == at_risk, f"{bid}: at_risk_slots wrong")
    utt = P["utterance_slots"]
    for k, v in utt.items():
        rep.check(k in required and gold.get(k) == v, f"{bid}: utterance_slots[{k}]={v!r} is not a gold value")
    rep.check(P["candidate_set_exhaustive"] is False, f"{bid}: open-world default requires candidate_set_exhaustive false")
    for k in required:  # the user only has to say what the utterance did not already settle
        if k not in utt or k in at_risk:
            rep.check(k in P["say"], f"{bid}: no literal user phrase for {k!r}")
    if cf is not None:
        rep.check(cf.get("kind") in CONFUSABLE_KINDS and bool(cf.get("competitors")), f"{bid}: bad confusable")
        rep.check(cf.get("affects") in required, f"{bid}: confusable affects non-required key")
        rep.check(str(cf.get("surface", "")).lower() in P["reference_text"].lower(), f"{bid}: confusable surface not in text")
    missing = [k for k in required if k not in utt]

    # ----- alternatives vs axis ----------------------------------------------------------------
    alts = P["alternatives"]
    rep.check(len(alts) >= (1 if cat == "none" else 2), f"{bid}: too few alternatives ({len(alts)})")
    fam = P["intent_families"]
    for a in alts:
        t = a["task"]
        rep.check(t != {"intent": gt["intent"], "slots": gt["slots"]} and bool(a.get("why")), f"{bid}: alternative equals gold or lacks why")
        rep.check(set(t["slots"]) == set(gt["slots"]), f"{bid}: alternative slot keys differ from gold (incomplete task)")
        for k, v in t["slots"].items():
            rep.check(valid_slot(k, v), f"{bid}: implausible/invalid alternative slot {k}={v!r}")
        diff = [k for k, v in t["slots"].items() if gt["slots"].get(k) != v] + (["intent"] if t["intent"] != gt["intent"] else [])
        rep.check(sorted(a.get("differs_in", [])) == sorted(diff), f"{bid}: differs_in {a.get('differs_in')} != {diff}")
        if cat == "underspecified":
            rep.check(t["intent"] == gt["intent"] and diff, f"{bid}: slot-axis alternatives must keep the intent")
        if cat == "intent":
            rep.check(t["intent"] != gt["intent"], f"{bid}: intent-axis alternative keeps the same intent")
        if cat == "acoustic":
            rep.check(cf is not None and cf["affects"] in diff, f"{bid}: acoustic alternative must differ in the confusable")
    if cat == "intent":
        rep.check(any(fam.get(a["task"]["intent"]) not in (None, fam.get(gt["intent"])) for a in alts),
                  f"{bid}: no intent alternative crosses an action family (looks like a slot problem)")
    if cat == "underspecified":
        rep.check(bool(missing), f"{bid}: underspecified item supplies every required slot")
        rep.check("intent" not in required, f"{bid}: underspecified item must have a clear intent")
    if cat == "none":
        rep.check(not missing, f"{bid}: 'none' item leaves {missing} unresolved")
    if cat == "intent":
        rep.check("intent" in required and "intent" not in utt, f"{bid}: intent item must leave intent unresolved")
    # ----- acceptable outcomes -----------------------------------------------------------------
    acc = P["acceptable_tasks"]
    rep.check(bool(acc) and any(matches({"intent": gt["intent"], "slots": gt["slots"]}, a) for a in acc),
              f"{bid}: acceptable_tasks must accept the gold task")
    wild = any("*" in a.get("slots", {}).values() for a in acc)
    rep.check(not wild or bool(P["acceptable_note"].strip()), f"{bid}: wildcard acceptance needs a justification note")
    rep.check(wild == bool(P["acceptable_note"].strip()), f"{bid}: acceptable_note present without a wildcard (or vice versa)")

    # ----- probe coverage -----------------------------------------------------------------------
    probes = P["probes"]
    pids = [p.get("probe_id") for p in probes]
    rep.check(len(set(pids)) == len(pids) and all(str(x).startswith(bid + ".") for x in pids), f"{bid}: probe ids")
    by = {"open": [], "closed": [], "confirm": []}
    for p in probes:
        by.setdefault(p.get("question_type"), []).append(p)
    conf_ps = by["confirm"]
    rep.check(any(p.get("proposal_is_correct") is True for p in conf_ps), f"{bid}: no correct confirmation proposal")
    rep.check(any(p.get("proposal_is_correct") is False for p in conf_ps), f"{bid}: no incorrect confirmation proposal")
    rtypes = {p["reply"]["type"] for p in conf_ps}
    rep.check({"yes", "no_bare", "no_correction"} <= rtypes, f"{bid}: confirmations must include yes, no_bare and no_correction")
    if cat == "none":
        rep.check(not by["open"] and not by["closed"], f"{bid}: 'none' items have no open/closed probes")
        rep.check(P["not_applicable_questions"] == ["open", "closed"], f"{bid}: not_applicable_questions")
    else:
        rep.check(bool(by["open"]), f"{bid}: no open probe")
        rep.check(any(p.get("options_contain_gold") for p in by["closed"]), f"{bid}: no gold-containing closed probe")
        rep.check(any(p.get("options_contain_gold") is False for p in by["closed"]), f"{bid}: no gold-omitting closed probe")
        rep.check(P["not_applicable_questions"] == [], f"{bid}: not_applicable_questions should be empty")
        if len(missing) >= 2:
            rep.check(any(p["recovery"]["strict"]["status"] == "partial" for p in by["open"]),
                      f"{bid}: several slots missing but no partial-coverage open probe")
    for p in probes:
        check_probe(bid, p, P, rep)


def validate(items_path, manifest_path, config_path, src_dir=None, author_path=None):
    rep = Report()
    snr_levels = yaml.safe_load(Path(config_path).read_text())["data"]["snr_levels"]
    items_path = Path(items_path)
    records = []
    for n, line in enumerate(items_path.read_text().splitlines(), 1):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            rep.check(False, f"line {n}: invalid JSON ({exc})")
    for r in records:
        check_record(r, snr_levels, rep)
    ok = [r for r in records if isinstance(r, dict) and set(r) == TOP_KEYS and set(r["private"]) == PRIVATE_KEYS]

    # ----- counts, uniqueness, leakage -------------------------------------------------------------
    rep.check(len(records) == 50, f"expected 50 base items, found {len(records)}")
    ids = [r.get("base_id") for r in records]
    rep.check(len(set(ids)) == len(ids), f"duplicate base_ids: {[i for i, c in Counter(ids).items() if c > 1]}")
    cell = Counter((r["private"]["ambiguity_type"], r["split"]) for r in ok)
    for cat, planned in APPROVED.items():
        for sp, n in zip(SPLITS, planned):
            rep.check(cell[(cat, sp)] == n, f"{cat}/{sp}: expected {n}, found {cell[(cat, sp)]}")
    tot = Counter(r["split"] for r in ok)
    rep.check([tot[s] for s in SPLITS] == [30, 10, 10], f"split totals {dict(tot)} != 30/10/10")
    vids = [v["variant_id"] for r in ok for v in r["agent_visible"]["audio_variants"]]
    rep.check(len(vids) == 50 * len(snr_levels) and len(set(vids)) == len(vids),
              f"expected {50 * len(snr_levels)} unique variant ids, found {len(vids)} ({len(set(vids))} unique)")
    uids = [v[:9] for v in vids]
    rep.check(len(set(uids)) == len(records), "opaque item ids collide between base items")
    owner: dict[str, str] = {}
    for r in ok:
        for v in r["agent_visible"]["audio_variants"]:
            rep.check(owner.setdefault(v["variant_id"], r["split"]) == r["split"], f"{v['variant_id']}: in more than one split")
    texts = Counter(r["private"]["reference_text"].strip().lower() for r in ok)
    rep.check(all(c == 1 for c in texts.values()), f"duplicate reference_text: {[t for t, c in texts.items() if c > 1]}")
    tasks = Counter(json.dumps(r["private"]["gold_task"]["slots"], sort_keys=True) + "|" + r["private"]["gold_task"]["intent"] for r in ok)
    rep.check(all(c == 1 for c in tasks.values()), "a gold task (intent+slots) is shared by more than one item")

    # ----- dataset-level fairness checks --------------------------------------------------------------
    allp = [p for r in ok for p in r["private"]["probes"]]
    pos = [p["option_order"]["gold_position"] for p in allp if p.get("question_type") == "closed" and p.get("options_contain_gold")]
    stats = {}
    if pos:
        first_share = pos.count(0) / len(pos)
        stats["gold_position_counts"] = dict(sorted(Counter(pos).items()))
        rep.check(first_share <= 0.5, f"gold option is first in {first_share:.0%} of closed questions (limit 50%)")
        rep.check(len(set(pos)) >= 2 and max(Counter(pos).values()) / len(pos) <= 0.5, "gold option position is not spread")
    words = lambda t: len(t.split())  # noqa: E731
    means = {}
    for key, sel in {"open": lambda p: p["question_type"] == "open", "closed_select": lambda p: p["reply"]["type"] == "select",
                     "yes": lambda p: p["reply"]["type"] == "yes", "no_correction": lambda p: p["reply"]["type"] == "no_correction"}.items():
        ws = [words(p["reply"]["text"]) for p in allp if sel(p)]
        means[key] = round(sum(ws) / len(ws), 2) if ws else None
    stats["mean_reply_words"] = means
    if means["open"] and means["closed_select"]:
        ratio = max(means["open"], means["closed_select"]) / min(means["open"], means["closed_select"])
        rep.check(ratio <= 2.5, f"open vs closed reply length differs by {ratio:.2f}x (limit 2.5x)")
    acoustic = [r for r in ok if r["private"]["ambiguity_type"] == "acoustic"]
    for r in acoustic:
        P = r["private"]
        key = P["acoustic_confusable"]["affects"]
        for p in P["probes"]:
            if key in p["reply"]["supplies"]:
                rep.check(has_phrase(p["reply"]["text"], P["say"][key]), f"{p['probe_id']}: replies must restate the confusable literally")

    # ----- manifest -----------------------------------------------------------------------------------------
    mpath = Path(manifest_path)
    if not rep.check(mpath.exists(), f"manifest not found: {mpath}"):
        return rep, stats, ok
    m = json.loads(mpath.read_text())
    rep.check(m.get("items_sha256") == hashlib.sha256(items_path.read_bytes()).hexdigest(), "manifest items_sha256 mismatch")
    rep.check(m.get("schema_version") == SCHEMA_VERSION, "manifest schema_version")
    for sp in SPLITS:
        rep.check(m.get("split_ids", {}).get(sp) == [r["base_id"] for r in ok if r["split"] == sp], f"manifest split_ids[{sp}] mismatch")
    rep.check(m.get("snr_levels_planned") == snr_levels, "manifest SNR levels differ from config")
    rep.check(m.get("planned_counts", {}).get("by_category_split") == {c: dict(zip(SPLITS, v)) for c, v in APPROVED.items()},
              "manifest planned counts differ from the approved allocation")
    pc = m.get("probe_counts", {})
    rep.check(pc.get("total") == len(allp), "manifest probe total mismatch")
    rep.check(pc.get("confirm_correct") == sum(1 for p in allp if p.get("proposal_is_correct") is True), "manifest confirm_correct mismatch")
    rep.check(pc.get("confirm_incorrect") == sum(1 for p in allp if p.get("proposal_is_correct") is False), "manifest confirm_incorrect mismatch")
    rc = m.get("recovery_counts_strict", {})
    for s in ("full", "partial", "none"):
        rep.check(rc.get(s) == sum(1 for p in allp if p["recovery"]["strict"]["status"] == s), f"manifest recovery count {s} mismatch")
    ap = Path(author_path) if author_path else ROOT / "scripts" / "author_phase1.py"
    if ap.exists():
        rep.check(m.get("author_script_sha256") == hashlib.sha256(ap.read_bytes()).hexdigest(),
                  "author script changed since the dataset was generated: regenerate before trusting this file")

    # ----- static scan of agent-side code (no modification of it) ---------------------------------------------
    if src_dir:
        for f in sorted(Path(src_dir).glob("*.py")):
            txt = f.read_text()
            for tok in PRIVATE_TOKENS:
                rep.check(tok not in txt, f"{f.name}: agent-side code references private field {tok!r}")
            for tok in LEGACY_TOKENS:
                if tok in txt:
                    rep.warnings.append(f"{f.name}: references legacy v1 field {tok!r}; a Phase 2 adapter must keep "
                                        "private data out of agent prompts")
    return rep, stats, ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--items", default=str(ROOT / "data/phase1/base_items.jsonl"))
    ap.add_argument("--manifest", default=str(ROOT / "data/phase1/manifest.json"))
    ap.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    ap.add_argument("--src", default=None, help="scan this directory for references to private fields")
    args = ap.parse_args()

    rep, stats, ok = validate(args.items, args.manifest, args.config, args.src)
    tot = Counter(r["split"] for r in ok)
    cell = Counter((r["private"]["ambiguity_type"], r["split"]) for r in ok)
    print(f"base items: {len(ok)}   split totals: " + "/".join(str(tot[s]) for s in SPLITS) + " (train/val/test)")
    print(f"{'category':<16}{'train':>6}{'val':>6}{'test':>6}{'total':>7}")
    for cat in APPROVED:
        row = [cell[(cat, sp)] for sp in SPLITS]
        print(f"{cat:<16}{row[0]:>6}{row[1]:>6}{row[2]:>6}{sum(row):>7}")
    for k, v in stats.items():
        print(f"{k}: {v}")
    for w in rep.warnings:
        print("WARN:", w)
    if rep.errors:
        print(f"\nFAIL: {len(rep.errors)} problem(s)")
        for e in rep.errors:
            print("  -", e)
        return 1
    print("\nOK: internal-consistency checks passed (this is not scientific validation)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
