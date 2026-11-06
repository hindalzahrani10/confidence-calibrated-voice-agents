"""Independent validator for the Phase 2 audio (audio.v1).

Shares no code with scripts/generate_phase2_audio.py or src/noise.py: it has its
own level/SNR implementation and rebuilds every noisy clip from the stored
clean clip + noise asset + offset + gain, so a generator bug cannot validate
itself. It checks consistency, not that the speech is natural or the labels
correct. All audio is synthetic TTS.

    python scripts/validate_phase2_audio.py [--root data/phase2/audio.v1]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

ROOT = Path(__file__).resolve().parent.parent
SCALE = 32767.0
INITIAL_ID = re.compile(r"^u[0-9a-f]{8}-[a-d]$")
REPLY_ID = re.compile(r"^r[0-9a-f]{10}$")
LEAK = ("none", "acoustic", "intent", "underspec", "clean", "snr", "train", "val", "test", "p1_")


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def frame_ms(x, sr, fm, hm):
    f, h = int(sr * fm / 1000), int(sr * hm / 1000)
    n = 1 + (len(x) - f) // h
    idx = np.arange(f)[None, :] + h * np.arange(n)[:, None]
    return (x.astype(np.float64)[idx] ** 2).mean(axis=1)


def active_power(x, sr, al):
    e = frame_ms(x, sr, al["frame_ms"], al["hop_ms"])
    return float(e[e > e.max() * 10 ** (-al["dynamic_range_db"] / 10)].mean())


def q16(x):
    return np.clip(np.rint(x * SCALE), -32768, 32767).astype(np.int16)


def validate(root: Path, config_path: Path):
    errors: list[str] = []
    ck = lambda ok, msg: ok or errors.append(msg)  # noqa: E731
    cfg = yaml.safe_load(config_path.read_text())
    levels = yaml.safe_load((ROOT / "configs/default.yaml").read_text())["data"]["snr_levels"]
    tol, al = cfg["tolerances"], cfg["active_level"]
    sr = cfg["format"]["sample_rate"]

    man = json.loads((root / "manifest.json").read_text())
    rows = [json.loads(l) for l in (root / "audio_manifest.jsonl").read_text().splitlines()]
    ck(man["audio_manifest_sha256"] == sha((root / "audio_manifest.jsonl").read_bytes()), "audio_manifest.jsonl hash != manifest.json")
    ck(man.get("synthetic") is True and all(r.get("synthetic") is True for r in rows), "dataset must be marked synthetic")
    ck(all(r["speaker"].startswith("tts:") for r in rows), "speaker must be identified as TTS")
    ck(man["config_sha256"] == sha(config_path.read_bytes()), "config changed since the audio was built")

    # ---- Phase 1 linkage (hash-checked snapshot) ---------------------------------------------------
    snap = ROOT / man["phase1_snapshot"]
    for line in (snap / "SHA256SUMS").read_text().splitlines():
        h, name = line.split(None, 1)
        ck(sha((snap / name.strip()).read_bytes()) == h, f"Phase 1 snapshot file {name.strip()} altered")
    recs = [json.loads(l) for l in (snap / "base_items.jsonl").read_text().splitlines()]
    ck(man["phase1_dataset_sha256"] == sha((snap / "base_items.jsonl").read_bytes()), "manifest Phase 1 hash mismatch")
    owner = {r["base_id"]: r for r in recs}

    # ---- identity, counts, isolation ---------------------------------------------------------------------
    ids = [r["clip_id"] for r in rows]
    ck(len(ids) == len(set(ids)), "duplicate clip ids")
    ck(len(rows) == man["counts"]["clips"], "manifest clip count mismatch")
    init = [r for r in rows if r["kind"] == "initial"]
    ck(len(init) == 50 * len(levels), f"expected {50 * len(levels)} initial clips, found {len(init)}")
    for r in recs:
        want = {c["variant_id"]: c["snr_db"] for c in r["private"]["conditions"]}
        got = {x["clip_id"]: x["snr_nominal_db"] for x in init if x["owner_base_id"] == r["base_id"]}
        ck(got == want, f"{r['base_id']}: initial clips/SNRs differ from Phase 1 conditions")
        vis = sorted(v["variant_id"] for v in r["agent_visible"]["audio_variants"])
        ck(sorted(got) == vis, f"{r['base_id']}: clip ids differ from agent-visible variant ids")
        # every Phase 1 reply must have audio at every SNR
        for p in r["private"]["probes"]:
            mine = [x for x in rows if x["owner_base_id"] == r["base_id"] and x["kind"] == "reply" and p["probe_id"] in x["used_by_probes"]]
            ck(sorted(x["snr_nominal_db"] is None and -1 or x["snr_nominal_db"] for x in mine) == sorted([-1] + [s for s in levels if s is not None] * cfg["replicates"]["reply"]),
               f"{p['probe_id']}: reply audio missing or incomplete across SNRs")
            ck(all(x["text"].strip().lower() == p["reply"]["text"].strip().lower() for x in mine), f"{p['probe_id']}: reply text mismatch")
    for r in rows:
        ck(r["split"] == owner[r["owner_base_id"]]["split"], f"{r['clip_id']}: split differs from its Phase 1 item")
        ck(r["path"].split("/")[-2] == r["split"], f"{r['clip_id']}: stored under the wrong split directory")
        ck(bool((INITIAL_ID if r["kind"] == "initial" else REPLY_ID).match(r["clip_id"].split("~")[0])), f"{r['clip_id']}: id is not opaque")
        ck(not any(w in r["path"].split("/")[-1].lower() for w in LEAK), f"{r['clip_id']}: file name leaks a label word")
        ck(not re.search(r"\d", r["text"]), f"{r['clip_id']}: digits in TTS text")
    splits_of = {}
    for r in rows:
        ck(splits_of.setdefault(r["clip_id"], r["split"]) == r["split"], f"{r['clip_id']}: in more than one split")
    for sp in ("train", "val", "test"):
        ck(man["counts"]["by_split"][sp] == sum(r["split"] == sp for r in rows), f"split count {sp} mismatch")
    # A duplicate waveform inside one item is acceptable only when an initial utterance and a reply have the
    # same case-folded text (TTS is case-insensitive); any other duplicate is unnecessary.
    groups: dict = {}
    for r in rows:
        groups.setdefault((r["owner_base_id"], r["sha256"]), []).append(r)
    for (own, _), g in groups.items():
        if len(g) > 1:
            benign = len({x["text"].strip().lower() for x in g}) == 1 and len({x["kind"] for x in g}) == len(g) == 2
            ck(benign, f"{own}: duplicate waveform ({[x['clip_id'] for x in g]})")
    summary_dupes = sum(len(g) > 1 for g in groups.values())

    # ---- per-file checks ------------------------------------------------------------------------------------
    noise_q, nsr = sf.read(ROOT / cfg["noise"]["path"], dtype="int16")
    ck(sha((ROOT / cfg["noise"]["path"]).read_bytes()) == cfg["noise"]["sha256"], "noise asset hash mismatch")
    noise = noise_q.astype(np.float32) / SCALE
    by_id = {r["clip_id"]: r for r in rows}
    pad = cfg["padding_s"]
    stats = Counter()
    snr_err = {s: [] for s in levels if s is not None}
    for r in rows:
        p = root / r["split"] / f"{r['clip_id']}.wav"
        if not p.exists():
            errors.append(f"{r['clip_id']}: file missing")
            continue
        raw = p.read_bytes()
        ck(sha(raw) == r["sha256"], f"{r['clip_id']}: file hash != manifest")
        info = sf.info(p)
        ck(info.samplerate == sr and info.channels == 1 and info.subtype == "PCM_16" and info.format == "WAV",
           f"{r['clip_id']}: wrong format ({info.samplerate} Hz, {info.channels} ch, {info.subtype})")
        ck(len(raw) == 44 + 2 * info.frames * info.channels, f"{r['clip_id']}: unexpected WAV header/metadata")
        if info.channels != 1 or info.samplerate != sr or info.subtype != "PCM_16":
            continue  # already reported; the checks below assume mono 16 kHz PCM16
        q, _ = sf.read(p, dtype="int16")
        x = q.astype(np.float64) / SCALE
        ck(bool(np.isfinite(x).all()), f"{r['clip_id']}: non-finite samples")
        ck(float(np.abs(x).max()) <= tol["peak_max"] and not np.any(np.abs(q.astype(np.int32)) >= 32767), f"{r['clip_id']}: clipping")
        ck(tol["duration_s"][0] <= info.frames / sr <= tol["duration_s"][1], f"{r['clip_id']}: duration {info.frames / sr:.2f}s out of range")
        ck(float(np.abs(x).max()) > 0.01, f"{r['clip_id']}: (near-)silent clip")
        if r["snr_nominal_db"] is None:
            lead, trail = int(pad["leading"] * sr), int(pad["trailing"] * sr)
            ck(not q[:lead].any() and not q[-trail:].any(), f"{r['clip_id']}: padding is not digital silence")
            lvl = 10 * np.log10(active_power(x, sr, al))
            ck(abs(lvl - cfg["speech_level_db"]) <= tol["level_db"], f"{r['clip_id']}: clean level {lvl:.2f} dB off target")
            stats["clean"] += 1
        else:
            src = by_id.get(r["clean_clip_id"])
            if src is None:
                errors.append(f"{r['clip_id']}: clean source missing")
                continue
            ck(src["owner_base_id"] == r["owner_base_id"] and src["split"] == r["split"] and src["snr_nominal_db"] is None,
               f"{r['clip_id']}: bad clean source")
            src_path = root / src["split"] / f"{src['clip_id']}.wav"
            if not src_path.exists():
                errors.append(f"{r['clip_id']}: clean source file missing")
                continue
            cq, _ = sf.read(src_path, dtype="int16")
            if cq.ndim != 1 or len(cq) != len(q):
                errors.append(f"{r['clip_id']}: clean source is not a same-length mono clip")
                continue
            n = r["noise"]
            lo, hi = (int(v * sr) for v in cfg["noise"]["partitions"][r["split"]])
            ck(lo <= n["offset_samples"] and n["offset_samples"] + len(q) <= hi, f"{r['clip_id']}: noise slice outside the {r['split']} partition")
            ck(n["partition"] == r["split"] and n["asset_sha256"] == cfg["noise"]["sha256"], f"{r['clip_id']}: bad noise provenance")
            seg = noise[n["offset_samples"]:n["offset_samples"] + len(q)]
            mix = q16(cq.astype(np.float64) / SCALE + n["gain"] * seg.astype(np.float64))
            lsb = int(np.abs(mix.astype(np.int32) - q.astype(np.int32)).max()) if len(mix) == len(q) else 99
            ck(lsb <= tol["reconstruct_lsb"], f"{r['clip_id']}: cannot be rebuilt from clean+noise+gain (max diff {lsb} LSB)")
            cx = cq.astype(np.float64) / SCALE
            achieved = 10 * np.log10(active_power(cx, sr, al) / np.mean((x - cx) ** 2))
            ck(abs(achieved - r["snr_nominal_db"]) <= tol["snr_db"], f"{r['clip_id']}: achieved SNR {achieved:.3f} vs nominal {r['snr_nominal_db']}")
            ck(abs(achieved - r["measured_snr_db_from_files"]) <= 0.01, f"{r['clip_id']}: manifest SNR differs from recomputed {achieved:.3f}")
            snr_err[r["snr_nominal_db"]].append(achieved - r["snr_nominal_db"])
            stats["noisy"] += 1
    summary = {"clips": len(rows), **stats, "benign_same_text_duplicates": summary_dupes,
               "snr_error_db": {str(k): (round(min(v), 4), round(max(v), 4)) for k, v in snr_err.items() if v}}
    return errors, summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=str(ROOT / "data/phase2/audio.v1"))
    ap.add_argument("--config", default=str(ROOT / "configs/phase2_audio.yaml"))
    a = ap.parse_args()
    errors, summary = validate(Path(a.root), Path(a.config))
    print(json.dumps(summary))
    if errors:
        print(f"FAIL: {len(errors)} problem(s)")
        for e in errors[:40]:
            print("  -", e)
        return 1
    print("OK: audio consistency checks passed (synthetic TTS speech; not a naturalness or label-correctness check)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
