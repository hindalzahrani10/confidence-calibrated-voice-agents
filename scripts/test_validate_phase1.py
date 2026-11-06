"""Negative tests for scripts/validate_phase1.py.

Each case corrupts a TEMPORARY copy of the dataset in one methodological way
and asserts the validator rejects it (and, where given, for the right reason).
The real dataset is only read, never written.

    TMPDIR=<scratch dir> python3 scripts/test_validate_phase1.py

Exits non-zero if the clean copy fails, or any corruption is not caught.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import validate_phase1 as V  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ITEMS = ROOT / "data/phase1/base_items.jsonl"
MANIFEST = ROOT / "data/phase1/manifest.json"
CONFIG = ROOT / "configs/default.yaml"


def rec(rs, name):
    return next(r for r in rs if r["base_id"] == f"p1_{name}")


def probe(r, qt, **where):
    for p in r["private"]["probes"]:
        if p["question_type"] == qt and all(p.get(k, p["reply"].get(k)) == v for k, v in where.items()):
            return p
    raise KeyError((r["base_id"], qt, where))


# ---- corruptions: (name, function(records), substring expected in some error) -----------------
def yes_on_incorrect(rs):
    probe(rec(rs, "acoustic_01"), "confirm", proposal_is_correct=False, type="no_bare")["reply"]["type"] = "yes"

def no_on_correct(rs):
    probe(rec(rs, "acoustic_01"), "confirm", proposal_is_correct=True)["reply"]["type"] = "no_bare"

def drop_incorrect_probes(rs):
    P = rec(rs, "intent_03")["private"]
    P["probes"] = [p for p in P["probes"] if p.get("proposal_is_correct") is not False]

def drop_correct_probe(rs):
    P = rec(rs, "intent_03")["private"]
    P["probes"] = [p for p in P["probes"] if p.get("proposal_is_correct") is not True]

def bare_no_full(rs):
    p = probe(rec(rs, "underspec_04"), "confirm", type="no_bare")
    p["recovery"]["strict"] = {"status": "full", "unresolved_slots": []}
    p["recovery"]["lenient"]["status"] = "full"; p["recovery"]["lenient"]["unresolved_slots"] = []

def bare_no_supplies(rs):
    probe(rec(rs, "acoustic_01"), "confirm", type="no_bare")["reply"]["supplies"] = {"contact": "Mom"}

def wrong_exclusion(rs):
    p = probe(rec(rs, "acoustic_01"), "confirm", type="no_bare")
    p["recovery"]["exclusions"][0]["joint"] = False

def gold_in_exclusion(rs):
    p = probe(rec(rs, "acoustic_01"), "closed", options_contain_gold=False)
    p["recovery"]["exclusions"][0]["values"].append("Mom")

def correction_text_missing_phrase(rs):
    probe(rec(rs, "acoustic_01"), "confirm", type="no_correction")["reply"]["text"] = "No, somebody else."

def wrong_unresolved(rs):
    p = probe(rec(rs, "underspec_04"), "confirm", type="no_correction")
    p["recovery"]["strict"]["unresolved_slots"] = []

def reply_value_not_gold(rs):
    p = probe(rec(rs, "acoustic_01"), "open")
    p["reply"]["supplies"]["contact"] = "Tom"

def gold_always_first(rs):
    for r in rs:
        for p in r["private"]["probes"]:
            if p["question_type"] == "closed" and p["options_contain_gold"]:
                i = p["option_order"]["gold_position"]
                for key in ("options", "option_values"):
                    p[key].insert(0, p[key].pop(i))
                p["option_order"]["gold_position"] = 0

def drop_gold_omitting(rs):
    P = rec(rs, "intent_05")["private"]
    P["probes"] = [p for p in P["probes"] if p.get("options_contain_gold") is not False]

def none_of_these_supplies_gold(rs):
    p = probe(rec(rs, "acoustic_01"), "closed", options_contain_gold=False)
    p["reply"]["supplies"] = {"contact": "Mom"}

def visible_id_leaks_category(rs):
    rec(rs, "none_01")["agent_visible"]["audio_variants"][0]["variant_id"] = "p1_none_01@clean"

def visible_extra_key(rs):
    rec(rs, "none_01")["agent_visible"]["gold_hint"] = "set_alarm"

def axis_mismatch(rs):
    rec(rs, "intent_01")["private"]["distinguishing_axis"] = "slot"

def intent_same_family(rs):
    P = rec(rs, "intent_01")["private"]
    for k in P["intent_families"]:
        P["intent_families"][k] = "one_family"

def intent_item_slot_alt(rs):
    a = rec(rs, "intent_01")["private"]["alternatives"][0]
    a["task"]["intent"] = rec(rs, "intent_01")["private"]["gold_task"]["intent"]

def invalid_gold_slot(rs):
    rec(rs, "acoustic_04")["private"]["gold_task"]["slots"]["level"] = 150

def invalid_alt_slot(rs):
    rec(rs, "underspec_12")["private"]["alternatives"][0]["task"]["slots"]["temperature_c"] = 50

def audio_path_set(rs):
    rec(rs, "none_03")["agent_visible"]["audio_variants"][0]["audio_path"] = "x.wav"

def split_leak(rs):
    # none_01 is train, intent_08 is val: a shared variant id would put one clip in two splits
    rec(rs, "none_01")["agent_visible"]["audio_variants"][0]["variant_id"] = \
        rec(rs, "intent_08")["agent_visible"]["audio_variants"][0]["variant_id"]

def move_item_between_splits(rs):
    rec(rs, "none_01")["split"] = "val"

def duplicate_base_id(rs):
    rec(rs, "none_02")["base_id"] = "p1_none_01"

def open_question_reveals_gold(rs):
    p = probe(rec(rs, "underspec_01"), "open")
    p["question"] = "Is seven in the morning right?"

def acceptable_excludes_gold(rs):
    rec(rs, "none_01")["private"]["acceptable_tasks"] = []

def exhaustive_candidate_set(rs):
    rec(rs, "acoustic_01")["private"]["candidate_set_exhaustive"] = True

def option_seed_missing(rs):
    probe(rec(rs, "acoustic_01"), "closed", options_contain_gold=True)["option_order"]["seed"] = ""

def provenance_external(rs):
    rec(rs, "none_01")["private"]["meta"]["external_dataset"] = "SLURP"

def fabricated_agreement(rs):
    rec(rs, "none_01")["private"]["annotation"]["agreement"] = 0.87

def missing_private_key(rs):
    del rec(rs, "none_01")["private"]["acceptable_tasks"]

def lenient_without_assumption(rs):
    p = probe(rec(rs, "underspec_04"), "confirm", type="no_correction")
    p["recovery"]["lenient"]["status"] = "none"

def wildcard_without_note(rs):
    rec(rs, "underspec_03")["private"]["acceptable_note"] = ""

def underspec_complete(rs):
    P = rec(rs, "underspec_01")["private"]
    P["utterance_slots"] = {"time": "07:00"}

CASES = [
    ("yes accepted for an incorrect proposal", yes_on_incorrect, "must receive a 'no'"),
    ("'no' given to a correct proposal", no_on_correct, "must receive 'yes'"),
    ("incorrect confirmation probe missing", drop_incorrect_probes, "no incorrect confirmation"),
    ("correct confirmation probe missing", drop_correct_probe, "no correct confirmation"),
    ("bare 'no' marked full recovery", bare_no_full, "can never be full"),
    ("bare 'no' supplies a value", bare_no_supplies, "must not supply"),
    ("rejected proposal not joint exclusion", wrong_exclusion, "joint exclusion"),
    ("gold listed as excluded option", gold_in_exclusion, "exactly the rejected options"),
    ("correction text lacks literal phrase", correction_text_missing_phrase, "lacks phrase"),
    ("unresolved slots wrong", wrong_unresolved, "unresolved"),
    ("reply value differs from gold", reply_value_not_gold, "!= gold"),
    ("gold option always first", gold_always_first, "gold option is first"),
    ("gold-omitting closed question missing", drop_gold_omitting, "no gold-omitting closed probe"),
    ("'none of these' supplies the gold", none_of_these_supplies_gold, "must not supply or confirm"),
    ("agent-visible id leaks category", visible_id_leaks_category, "not opaque"),
    ("extra agent-visible field", visible_extra_key, "only audio_variants"),
    ("axis inconsistent with category", axis_mismatch, "inconsistent with category"),
    ("intent item with no cross-family alternative", intent_same_family, "action family"),
    ("intent item with a slot-only alternative", intent_item_slot_alt, "keeps the same intent"),
    ("invalid gold slot value", invalid_gold_slot, "implausible"),
    ("invalid alternative slot value", invalid_alt_slot, "implausible"),
    ("audio path not null", audio_path_set, "audio must be null"),
    ("variant id shared across splits", split_leak, "more than one split"),
    ("category/split counts off", move_item_between_splits, "expected"),
    ("duplicate base id", duplicate_base_id, "duplicate base_ids"),
    ("open question reveals unresolved value", open_question_reveals_gold, "question reveals"),
    ("acceptable_tasks rejects the gold", acceptable_excludes_gold, "must accept the gold"),
    ("candidate set declared exhaustive", exhaustive_candidate_set, "open-world"),
    ("option-order seed missing", option_seed_missing, "seed"),
    ("external provenance claimed", provenance_external, "provenance"),
    ("fabricated agreement score", fabricated_agreement, "unfabricated"),
    ("required private field missing", missing_private_key, "private keys differ"),
    ("lenient differs without an assumption", lenient_without_assumption, "lenient"),
    ("wildcard acceptance without note", wildcard_without_note, "wildcard"),
    ("underspecified item fully specified", underspec_complete, "supplies every required slot"),
]


def run_case(name, fn, expect, base, tmp):
    rs = copy.deepcopy(base)
    fn(rs)
    items = tmp / "items.jsonl"
    items.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rs))
    man = json.loads(MANIFEST.read_text())
    man["items_sha256"] = hashlib.sha256(items.read_bytes()).hexdigest()  # isolate the intended defect
    (tmp / "manifest.json").write_text(json.dumps(man))
    rep, _, _ = V.validate(items, tmp / "manifest.json", CONFIG)
    caught = bool(rep.errors)
    right_reason = any(expect in e for e in rep.errors)
    return caught, right_reason, rep.errors


def main() -> int:
    base = [json.loads(l) for l in ITEMS.read_text().splitlines()]
    failures = 0
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        caught, _, errs = run_case("clean copy", lambda rs: None, "", base, tmp)
        print(f"{'FAIL' if caught else 'ok  '} clean copy passes ({len(errs)} errors)")
        failures += bool(caught)
        for name, fn, expect in CASES:
            caught, reason, errs = run_case(name, fn, expect, base, tmp)
            status = "ok  " if caught and reason else "FAIL"
            failures += status == "FAIL"
            note = "" if status == "ok  " else f"   caught={caught} right_reason={reason} first={errs[:1]}"
            print(f"{status} {name}{note}")
        # manifest integrity: edit the items file after the manifest was written
        items = tmp / "items.jsonl"
        items.write_text(ITEMS.read_text().replace("p1_none_01", "p1_none_01", 1) + "\n")
        (tmp / "manifest.json").write_text(MANIFEST.read_text())
        rep, _, _ = V.validate(items, tmp / "manifest.json", CONFIG)
        hit = any("items_sha256" in e for e in rep.errors)
        print(f"{'ok  ' if hit else 'FAIL'} manifest hash detects an edited items file")
        failures += not hit
        # static scan: agent-side code referencing a private field
        src = tmp / "src"
        src.mkdir()
        (src / "agent.py").write_text("def act(item):\n    return item['gold_task']\n")
        (tmp / "items.jsonl").write_text(ITEMS.read_text())
        man = json.loads(MANIFEST.read_text())
        (tmp / "manifest.json").write_text(json.dumps(man))
        rep, _, _ = V.validate(tmp / "items.jsonl", tmp / "manifest.json", CONFIG, src_dir=src)
        hit = any("private field" in e for e in rep.errors)
        print(f"{'ok  ' if hit else 'FAIL'} static scan flags agent code reading a private field")
        failures += not hit
    total = len(CASES) + 3
    print(f"\n{total - failures}/{total} negative tests behaved as required")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
