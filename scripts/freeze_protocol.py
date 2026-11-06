"""Freeze the evaluation protocol BEFORE the held-out test split is opened.

Writes reports/protocol_freeze.json with hashes of everything that determines the result. After this file
exists, scripts may read the test split; changing any listed file afterwards invalidates the freeze (verified
by `python scripts/freeze_protocol.py --verify`). Refuses to freeze if development tuning artefacts are missing.
"""
import hashlib, json, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
FILES = ["src/ontology.py", "src/channel.py", "src/nmvoi.py", "src/simulator.py", "src/audio_service.py", "src/evalutil.py",
         "scripts/run_episodes.py", "scripts/analyze_results.py", "scripts/tune_policies.py", "scripts/estimate_channel.py",
         "scripts/fit_snr_estimator.py", "scripts/phase3_task_inference.py", "scripts/build_catalogue.py",
         "configs/default.yaml", "configs/phase2_audio.yaml", "data/phase3/catalogue.v1.json", "data/phase3/calibration.v1.json",
         "data/phase4/channel.v1.json", "data/phase4/snr_estimator.v1.json", "data/phase5/tuned.json",
         "data/phase2/audio.v1/audio_manifest.jsonl", "data/phase3/asr.v1/config.json"]
sha = lambda p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()  # noqa: E731
out = ROOT / "reports/protocol_freeze.json"
if "--verify" in sys.argv:
    fr = json.loads(out.read_text()); bad = [p for p, h in fr["files"].items() if sha(p) != h]
    print("FREEZE INTACT" if not bad else f"FREEZE BROKEN: {bad}"); sys.exit(1 if bad else 0)
missing = [p for p in FILES if not (ROOT / p).exists()]
if missing:
    raise SystemExit(f"cannot freeze; missing: {missing}")
if out.exists():
    raise SystemExit("protocol already frozen; refusing to overwrite")
costs = "src/nmvoi.py: Costs(open=0.30, closed=0.20, confirm=0.10); reward +1/-1/0"
out.write_text(json.dumps({"files": {p: sha(p) for p in FILES}, "costs": costs,
                           "primary_metrics": ["task success rate", "mean clarification turns", "mean utility (reward - interaction cost)"],
                           "test_split_policy": "opened once, after this file was written; no tuning on it",
                           "note": "human semantic review of the dataset is deferred; all data synthetic"}, indent=1) + "\n")
print("protocol frozen ->", out)
