"""Validate the completed human review workbook against the experimental checklist and merge the H_ columns.

Read-only on both inputs (stdlib only: the .xlsx is parsed with zipfile/xml). Merges nothing if a SERIOUS
problem is found (missing/duplicate/extra IDs, any change to an experimental column, header mismatch).
Human judgments are copied verbatim as strings; nothing is imputed, corrected or normalised.

    python scripts/merge_human_review.py [--dir reports/audit/review_package]
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
M = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
H_COLS = ["H_question_relevance(1-5 or Y/N)", "H_reply_appropriateness", "H_audio_asr_consistency",
          "H_final_action_correct", "H_overall_valid(Y/N/unsure)", "H_notes", "H_reviewed_by_human"]


def col_index(ref: str) -> int:
    n = 0
    for ch in re.match(r"[A-Z]+", ref).group():
        n = n * 26 + ord(ch) - 64
    return n - 1


def read_xlsx(path: Path, sheet: str = "Dialogue Review"):
    z = zipfile.ZipFile(path)
    ss = ["".join(t.text or "" for t in si.iter(M + "t")) for si in ET.fromstring(z.read("xl/sharedStrings.xml")).iter(M + "si")]
    wb = ET.fromstring(z.read("xl/workbook.xml"))
    names = [s.get("name") for s in wb.find(M + "sheets")]
    idx = names.index(sheet) + 1
    rows = []
    for row in ET.fromstring(z.read(f"xl/worksheets/sheet{idx}.xml")).iter(M + "row"):
        r = {}
        for c in row.iter(M + "c"):
            t, v = c.get("t"), c.find(M + "v")
            if t == "s":
                val = ss[int(v.text)]
            elif t == "inlineStr":
                val = "".join(x.text or "" for x in c.iter(M + "t"))
            else:
                val = v.text if v is not None else ""
            r[col_index(c.get("r"))] = val if val is not None else ""
        rows.append(r)
    header = [rows[0].get(i, "") for i in range(max(rows[0]) + 1)]
    data = [{header[i]: r.get(i, "") for i in range(len(header))} for r in rows[1:]]
    return header, [d for d in data if any(str(v).strip() for v in d.values())], names


def same(a: str, b: str) -> bool:
    if a == b:
        return True
    try:
        return float(a) == float(b)
    except ValueError:
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="reports/audit/review_package")
    d = ROOT / ap.parse_args().dir
    csv_path, xlsx_path = d / "review_checklist.csv", d / "dialogue_review_20(1).xlsx"
    out_csv, out_md, out_json = d / "review_checklist_human_completed.csv", d / "human_review_validation_report.md", d / "human_review_validation.json"
    for p in (out_csv, out_md, out_json):
        if p.exists():
            raise SystemExit(f"refusing to overwrite {p}")
    serious, warn, info = [], [], []
    with open(csv_path, newline="") as fh:
        rd = csv.DictReader(fh)
        c_header, c_rows = rd.fieldnames, list(rd)
    x_header, x_rows, sheets = read_xlsx(xlsx_path)
    info.append(f"readable: csv {len(c_rows)} rows x {len(c_header)} cols; xlsx sheets {sheets}, {len(x_rows)} rows x {len(x_header)} cols")

    expect = [f"D{i:02d}" for i in range(1, 21)]
    for name, rows in (("csv", c_rows), ("xlsx", x_rows)):
        ids = [r["dialogue_id"] for r in rows]
        dup = [k for k, n in Counter(ids).items() if n > 1]
        if dup:
            serious.append(f"{name}: duplicate dialogue_id {dup}")
        if sorted(set(ids)) != expect:
            serious.append(f"{name}: ids != D01..D20 (missing {sorted(set(expect) - set(ids))}, extra {sorted(set(ids) - set(expect))})")
        if len(ids) != 20:
            serious.append(f"{name}: {len(ids)} rows, expected 20")
    if c_header != x_header:
        serious.append(f"header mismatch: csv {c_header} vs xlsx {x_header}")
    if [c for c in c_header if c.startswith("H_")] != H_COLS:
        serious.append("unexpected H_ columns in csv")

    exp_cols = [c for c in c_header if not c.startswith("H_")]
    xmap = {r["dialogue_id"]: r for r in x_rows}
    mism = []
    for r in c_rows:
        x = xmap.get(r["dialogue_id"])
        if not x:
            continue
        for c in exp_cols:
            if not same(r[c], x.get(c, "")):
                mism.append((r["dialogue_id"], c, r[c], x.get(c, "")))
    if mism:
        serious.append(f"{len(mism)} experimental-column discrepancies: {mism[:10]}")
    # CSV H_ columns should still be blank (the 'original') -- else it is not the pristine checklist
    prefilled = [(r["dialogue_id"], c) for r in c_rows for c in H_COLS if c != "H_reviewed_by_human" and r[c].strip()]
    if prefilled or any(r["H_reviewed_by_human"] != "NO" for r in c_rows):
        warn.append(f"CSV is not blank in its H_ columns: {prefilled[:5]}")
    # cross-check experimental columns against the logged dev episodes (read-only)
    eps = {}
    for l in (ROOT / "data/phase5/dev.v1/episodes.jsonl").read_text().splitlines():
        e = json.loads(l)
        eps[f"{e['agent']['policy']}|{e['agent']['clip_id']}"] = e
    for r in c_rows:
        e = eps.get(r["episode_key"])
        if e is None:
            serious.append(f"{r['dialogue_id']}: episode_key not in dev episodes")
            continue
        a, v = e["agent"], e["evaluator"]
        dec = f"ask_{a['question']['type']}" if a["action"] == "ask" else a["action"]
        if (dec, a["final"], f"{v['utility']:.1f}", v["base_id"], v["split"]) != (r["decision"], r["final"], f"{float(r['utility']):.1f}", r["base_id"], r["split"]):
            serious.append(f"{r['dialogue_id']}: checklist disagrees with logged episode")
    info.append("experimental columns re-checked against data/phase5/dev.v1/episodes.jsonl (read-only)")

    if serious:
        report = ["# Human-review validation (MERGE NOT PERFORMED)", "", "## Serious problems", *[f"- {s}" for s in serious]]
        out_md.write_text("\n".join(report) + "\n")
        print("\n".join(report))
        return 1

    # ---------------------------------------------------------------- inconsistencies (reported, never edited)
    QC, RC, AC, FC, OC, NC, RV = H_COLS
    for r in c_rows:
        h = xmap[r["dialogue_id"]]
        did, asked = r["dialogue_id"], r["decision"].startswith("ask_")
        q, rp = h[QC].strip(), h[RC].strip()
        if asked and (not q or not rp):
            warn.append(f"{did}: clarification asked but question/reply evaluation blank (q='{q}', reply='{rp}')")
        if not asked and (q or rp):
            warn.append(f"{did}: no clarification asked but question/reply evaluation filled (q='{q}', reply='{rp}')")
        if (r["reply_audio"] != "") != asked:
            warn.append(f"{did}: reply_audio presence disagrees with decision")
        for c in (AC, FC, OC, NC, RV):
            if not h[c].strip():
                warn.append(f"{did}: {c} blank")
        if h[RV] != "YES":
            warn.append(f"{did}: H_reviewed_by_human = '{h[RV]}'")
        if q and not (re.fullmatch(r"[1-5]", q) or q in ("Y", "N", "unsure")):
            warn.append(f"{did}: question evaluation '{q}' outside 1-5/Y/N/unsure")
        for c in (RC, AC, FC, OC):
            if h[c].strip() and h[c] not in ("Y", "N", "unsure"):
                warn.append(f"{did}: {c} = '{h[c]}' outside Y/N/unsure")
        flags = set(f.strip().split(" ")[0] for f in r["auto_flags"].split(";") if f.strip())
        if h[AC] == "Y" and flags & {"F1", "F8"}:
            warn.append(f"{did}: audio/ASR marked Y although automatic flags {sorted(flags & {'F1','F8'})} say a transcript differs from intended (check)")
        if h[AC] == "N" and not flags & {"F1", "F8"}:
            warn.append(f"{did}: audio/ASR marked N but no automatic transcript-difference flag")
        if h[OC] == "Y" and h[FC] == "N":
            warn.append(f"{did}: overall valid=Y but final action correct=N (may be intended: dialogue valid, agent wrong)")
        if h[OC] == "N" and h[FC] == "Y":
            warn.append(f"{did}: overall valid=N but final action correct=Y")
        if h[OC] == "unsure" or h[FC] == "unsure" or h[AC] == "unsure":
            info.append(f"{did}: contains 'unsure' (OC={h[OC]}, FC={h[FC]}, AC={h[AC]}); kept verbatim")
        if len(h[NC]) > 0 and h[NC].rstrip()[-1] not in ".\"')":
            warn.append(f"{did}: H_notes may be truncated (ends '{h[NC][-20:]}')")

    merged = []
    for r in c_rows:
        row = dict(r)
        for c in H_COLS:
            row[c] = xmap[r["dialogue_id"]][c]
        merged.append(row)
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=c_header)
        w.writeheader()
        w.writerows(merged)
    # verify merge fidelity
    back = {r["dialogue_id"]: r for r in csv.DictReader(open(out_csv, newline=""))}
    assert all(back[i][c] == xmap[i][c] for i in expect for c in H_COLS)
    assert all(back[r["dialogue_id"]][c] == r[c] for r in c_rows for c in exp_cols)

    stats = {c: dict(Counter(xmap[i][c] if c != NC else "(text)" for i in expect)) for c in H_COLS}
    asked_n = sum(r["decision"].startswith("ask_") for r in c_rows)
    out_json.write_text(json.dumps({"serious": serious, "warnings": warn, "info": info, "distributions": stats,
                                    "n_asked": asked_n, "n_not_asked": 20 - asked_n}, indent=1) + "\n")
    md = ["# Human-review merge: validation report", "",
          f"Inputs (unmodified): `{csv_path.relative_to(ROOT)}`, `{xlsx_path.relative_to(ROOT)}`.  Output: `{out_csv.relative_to(ROOT)}`.", "",
          "## Checks", "- both files read; 20 unique IDs D01-D20 in each; no duplicates, no missing/extra IDs",
          f"- {len(exp_cols)} experimental columns compared cell-by-cell: **{len(mism)} discrepancies**; headers identical",
          "- experimental columns also match the logged dev episodes (decision, final, utility, base_id, split)",
          f"- 7 H_ columns copied verbatim for 20 dialogues; round-trip check of merged file passed",
          f"- dialogues with clarification: {asked_n}; without: {20 - asked_n}", "", "## Distributions of human entries (verbatim)"]
    for c in H_COLS[:-2] + H_COLS[-1:]:
        md.append(f"- {c}: {stats[c]}")
    md += ["", "## Items for human attention (nothing was altered)", *([f"- {w}" for w in warn] or ["- none"]), "",
           "## Information", *[f"- {i}" for i in info]]
    out_md.write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
