"""Create an immutable, versioned snapshot of the Phase 1 dataset.

Refuses to run unless the independent validator passes. The snapshot holds the
dataset, its manifest, the exact scripts and config that produced/validated it,
and SHA-256 checksums, in a directory named after the dataset hash. An existing
snapshot is never overwritten; re-running on unchanged data only re-verifies it.

A snapshot is NOT a freeze of the evaluation set (hard rule 3) and carries no
human validation: annotations remain an unreviewed AI draft.

    python3 scripts/snapshot_phase1.py
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VERSION = "phase1.v2.0"
SOURCES = {  # snapshot name -> repo path
    "base_items.jsonl": "data/phase1/base_items.jsonl",
    "manifest.json": "data/phase1/manifest.json",
    "code/author_phase1.py": "scripts/author_phase1.py",
    "code/validate_phase1.py": "scripts/validate_phase1.py",
    "code/test_validate_phase1.py": "scripts/test_validate_phase1.py",
    "code/snapshot_phase1.py": "scripts/snapshot_phase1.py",
    "config/default.yaml": "configs/default.yaml",
    "docs/phase1_dataset_spec.md": "notes/phase1_dataset_spec.md",
    "docs/phase1_human_review.md": "notes/phase1_human_review.md",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(cmd: list[str]) -> tuple[int, str]:
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


def git(*args: str) -> str:
    code, out = run(["git", *args])
    return out.strip() if code == 0 else "unavailable"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-root", default=str(ROOT / "data/phase1/snapshots"))
    args = ap.parse_args()

    code, out = run([sys.executable, "scripts/validate_phase1.py", "--src", "src"])
    if code != 0:
        print(out)
        print("Validation FAILED: no snapshot created.")
        return 1
    summary = [l for l in out.splitlines() if l.startswith(("base items", "OK"))]

    items_hash = sha256(ROOT / SOURCES["base_items.jsonl"])
    dest = Path(args.out_root) / f"{VERSION}+{items_hash[:12]}"
    if dest.exists():
        bad = [n for n, src in SOURCES.items()
               if n in ("base_items.jsonl", "manifest.json") and sha256(dest / n) != sha256(ROOT / src)]
        print(f"Snapshot already exists: {dest}\n" + ("MISMATCH in " + ", ".join(bad) if bad else "Contents verified unchanged."))
        return 1 if bad else 0

    dest.mkdir(parents=True)
    sums = {}
    for name, src in SOURCES.items():
        target = dest / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / src).read_bytes())
        sums[name] = sha256(target)
    (dest / "SHA256SUMS").write_text("".join(f"{h}  {n}\n" for n, h in sorted(sums.items())))

    info = {
        "snapshot_version": VERSION,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "dataset_sha256": items_hash,
        "provenance": "synthetic: 50 hand-authored base items; no external dataset",
        "annotation_status": "ai_drafted_unreviewed; no human review, no agreement score",
        "frozen_for_evaluation": False,
        "validator_summary": summary,
        "validator_note": "internal-consistency checks only; not scientific validation",
        "git_head": git("rev-parse", "HEAD"),
        "git_working_tree_clean": git("status", "--porcelain", "--", "scripts", "configs", "src", "notes") == "",
        "python": sys.version.split()[0],
        "note": "scripts were uncommitted when snapshotted; the copies under code/ are the authoritative generators",
        "files": sums,
    }
    (dest / "SNAPSHOT.json").write_text(json.dumps(info, indent=2, sort_keys=True) + "\n")
    for p in sorted(dest.rglob("*")):  # make it read-only
        os.chmod(p, 0o555 if p.is_dir() else 0o444)
    os.chmod(dest, 0o555)
    print(f"Created read-only snapshot: {dest}")
    print("\n".join(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
