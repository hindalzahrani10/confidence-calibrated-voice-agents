"""Generate the Phase 2 audio: TTS speech + controlled noisy conditions.

Reads the immutable Phase 1 snapshot (hash-checked), never the working copy,
and writes a NEW versioned directory; an existing version is never
overwritten (build happens in `<version>.partial`, then an atomic rename).

    python scripts/generate_phase2_audio.py              # build
    python scripts/generate_phase2_audio.py --verify 25  # regenerate 25 random clips, compare hashes

All speech is synthetic TTS. Gold labels never enter the audio path; the
agent-facing identifiers are the opaque ids already fixed in Phase 1.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import noise as N  # noqa: E402
from src.tts import PiperTTS  # noqa: E402


def sha_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_cfg(path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text())
    cfg["snr_levels"] = yaml.safe_load((ROOT / "configs/default.yaml").read_text())["data"]["snr_levels"]
    cfg["master_seed"] = yaml.safe_load((ROOT / "configs/default.yaml").read_text())["seed"]
    return cfg


def load_snapshot(cfg: dict) -> list[dict]:
    snap = ROOT / cfg["phase1_snapshot"]
    for line in (snap / "SHA256SUMS").read_text().splitlines():
        h, name = line.split(None, 1)
        if sha_file(snap / name.strip()) != h:
            raise SystemExit(f"Phase 1 snapshot file {name.strip()} does not match SHA256SUMS: refusing to build")
    return [json.loads(l) for l in (snap / "base_items.jsonl").read_text().splitlines()]


def plan_clips(records: list[dict], cfg: dict) -> list[dict]:
    """Every clip to build. Clean clips are listed once; noisy clips point at their clean source."""
    seed, levels = cfg["master_seed"], cfg["snr_levels"]
    plan: list[dict] = []
    for r in records:
        bid, split, P = r["base_id"], r["split"], r["private"]
        # --- initial utterance: ids were fixed (opaque) in Phase 1
        conds = {c["snr_db"]: c["variant_id"] for c in P["conditions"]}
        clean_id = conds[None]
        plan.append(dict(clip_id=clean_id, kind="initial", split=split, owner=bid, text=P["reference_text"],
                         snr=None, rep=0, clean_id=clean_id, variant_id=clean_id, used_by=[]))
        for snr in levels:
            if snr is None:
                continue
            for rep in range(cfg["replicates"]["initial"]):
                cid = conds[snr] if rep == 0 else f"{conds[snr]}~{rep}"
                plan.append(dict(clip_id=cid, kind="initial", split=split, owner=bid, text=P["reference_text"],
                                 snr=snr, rep=rep, clean_id=clean_id, variant_id=conds[snr], used_by=[]))
        # --- clarification replies: one clip per distinct reply text per item
        groups: dict[str, dict] = {}
        for p in P["probes"]:
            t = p["reply"]["text"].strip()
            g = groups.setdefault(t.lower(), {"text": t, "probes": []})
            g["probes"].append(p["probe_id"])
        for key, g in groups.items():
            rkey = hashlib.sha256(f"{bid}|{key}".encode()).hexdigest()[:8]
            cid = "r" + hashlib.sha256(f"{seed}|reply|{bid}|{rkey}|clean".encode()).hexdigest()[:10]
            plan.append(dict(clip_id=cid, kind="reply", split=split, owner=bid, text=g["text"], snr=None, rep=0,
                             clean_id=cid, variant_id=None, used_by=g["probes"], reply_key=rkey))
            for snr in levels:
                if snr is None:
                    continue
                for rep in range(cfg["replicates"]["reply"]):
                    nid = "r" + hashlib.sha256(f"{seed}|reply|{bid}|{rkey}|{snr}|{rep}".encode()).hexdigest()[:10]
                    plan.append(dict(clip_id=nid, kind="reply", split=split, owner=bid, text=g["text"], snr=snr,
                                     rep=rep, clean_id=cid, variant_id=None, used_by=g["probes"], reply_key=rkey))
    ids = [c["clip_id"] for c in plan]
    assert len(ids) == len(set(ids)), "clip id collision"
    return plan


class Builder:
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.sr = cfg["format"]["sample_rate"]
        self.al = cfg["active_level"]
        for key, path_key in (("model_sha256", "model"),):
            if sha_file(ROOT / cfg["tts"][path_key]) != cfg["tts"][key]:
                raise SystemExit("TTS model hash does not match config")
        if sha_file(ROOT / (cfg["tts"]["model"] + ".json")) != cfg["tts"]["model_config_sha256"]:
            raise SystemExit("TTS model config hash does not match config")
        if sha_file(ROOT / cfg["noise"]["path"]) != cfg["noise"]["sha256"]:
            raise SystemExit("noise asset hash does not match config")
        self.tts = PiperTTS(ROOT / cfg["tts"]["model"], cfg["tts"], self.sr)
        q, nsr = sf.read(ROOT / cfg["noise"]["path"], dtype="int16")
        assert nsr == self.sr and q.ndim == 1
        self.noise = N.from_pcm16(q)

    def clean(self, text: str):
        """(int16 padded clean clip, info)."""
        x = self.tts.synthesize(text)
        x, gain = N.normalise_speech(x, self.sr, self.cfg["speech_level_db"], self.al)
        pad = self.cfg["padding_s"]
        x = np.concatenate([np.zeros(int(pad["leading"] * self.sr), np.float32), x,
                            np.zeros(int(pad["trailing"] * self.sr), np.float32)])
        return N.to_pcm16(x), {"tts_gain": gain}

    def noisy(self, clean_q: np.ndarray, split: str, clip_id: str, snr: float):
        clean = N.from_pcm16(clean_q)
        seed = N.derive_seed(self.cfg["master_seed"], "noise", clip_id)
        off = N.draw_offset(len(clean), self.sr, self.cfg["noise"]["partitions"][split], seed)
        seg = self.noise[off:off + len(clean)]
        mix, gain, snr_float = N.mix_at_snr(clean, seg, snr, self.sr, self.al)
        q = N.to_pcm16(mix)
        # achieved SNR from the stored (quantised) components
        achieved_noise = N.from_pcm16(q).astype(np.float64) - clean.astype(np.float64)
        p_sp = N.active_ms(clean, self.sr, self.al["frame_ms"], self.al["hop_ms"], self.al["dynamic_range_db"])
        snr_file = 10.0 * np.log10(p_sp / float(np.mean(achieved_noise**2)))
        return q, {"noise_offset_samples": off, "noise_gain": gain, "noise_seed": seed,
                   "measured_snr_db": snr_float, "measured_snr_db_from_files": float(snr_file)}


def describe(q: np.ndarray, cfg: dict) -> dict:
    x = N.from_pcm16(q)
    al = cfg["active_level"]
    return {"n_samples": int(len(q)), "duration_s": round(len(q) / cfg["format"]["sample_rate"], 4),
            "active_level_db": round(N.level_db(N.active_ms(x, cfg["format"]["sample_rate"], al["frame_ms"],
                                                           al["hop_ms"], al["dynamic_range_db"])), 3),
            "peak_abs": round(float(np.abs(x).max()), 5), "clipped_samples": int(np.sum(np.abs(q.astype(np.int32)) >= 32767))}


def write_wav(path: Path, q: np.ndarray, sr: int) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, q, sr, subtype="PCM_16")
    return sha_file(path)


def versions() -> dict:
    import importlib.metadata as md
    return {k: md.version(k) for k in ("piper-tts", "onnxruntime", "numpy", "scipy", "soundfile")} | {
        "python": platform.python_version(), "platform": platform.platform()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "configs/phase2_audio.yaml"))
    ap.add_argument("--verify", type=int, default=0, help="regenerate N random clips and compare to the manifest")
    args = ap.parse_args()
    cfg = load_cfg(Path(args.config))
    records = load_snapshot(cfg)
    plan = plan_clips(records, cfg)
    out_root = ROOT / cfg["out_root"]
    final, partial = out_root / cfg["version"], out_root / (cfg["version"] + ".partial")
    sr = cfg["format"]["sample_rate"]
    builder = Builder(cfg)

    if args.verify:
        rows = {json.loads(l)["clip_id"]: json.loads(l) for l in (final / "audio_manifest.jsonl").read_text().splitlines()}
        pick = random.Random(0).sample(plan, min(args.verify, len(plan)))
        bad = 0
        cache: dict[str, np.ndarray] = {}
        for c in pick:
            if c["clean_id"] not in cache:
                cache[c["clean_id"]] = builder.clean(next(p for p in plan if p["clip_id"] == c["clean_id"])["text"])[0]
            q = cache[c["clean_id"]] if c["snr"] is None else builder.noisy(cache[c["clean_id"]], c["split"], c["clip_id"], c["snr"])[0]
            h = hashlib.sha256((ROOT / rows[c["clip_id"]]["path"]).read_bytes()).hexdigest()
            tmp = out_root / "_verify.wav"
            sf.write(tmp, q, sr, subtype="PCM_16")
            same = sha_file(tmp) == h
            tmp.unlink()
            bad += not same
        print(f"determinism check: {len(pick) - bad}/{len(pick)} regenerated clips are byte-identical")
        return 1 if bad else 0

    if final.exists():
        print(f"{final} already exists; versions are immutable. Bump `version` in the config to rebuild.")
        return 1
    if partial.exists():
        shutil.rmtree(partial)
    t0 = time.time()
    clean_cache: dict[str, np.ndarray] = {}
    rows: list[dict] = []
    for c in plan:
        split_dir = partial / c["split"]
        if c["snr"] is None:
            q, info = builder.clean(c["text"])
            clean_cache[c["clip_id"]] = q
            extra = {"tts": {"engine": "piper", "voice": "en_US-lessac-medium", "model_sha256": cfg["tts"]["model_sha256"],
                             "params": {k: cfg["tts"][k] for k in ("length_scale", "noise_scale", "noise_w_scale", "normalize_audio")},
                             "tts_gain": info["tts_gain"]},
                     "text_sha256": hashlib.sha256(c["text"].encode()).hexdigest(), "noise": None,
                     "measured_snr_db": None, "measured_snr_db_from_files": None}
        else:
            q, ninfo = builder.noisy(clean_cache[c["clean_id"]], c["split"], c["clip_id"], c["snr"])
            extra = {"tts": None, "text_sha256": hashlib.sha256(c["text"].encode()).hexdigest(),
                     "noise": {"asset_id": cfg["noise"]["asset_id"], "asset_sha256": cfg["noise"]["sha256"],
                               "partition": c["split"], "offset_samples": ninfo["noise_offset_samples"],
                               "gain": ninfo["noise_gain"], "seed": ninfo["noise_seed"]},
                     "measured_snr_db": ninfo["measured_snr_db"], "measured_snr_db_from_files": ninfo["measured_snr_db_from_files"]}
        rel = f"{cfg['out_root']}/{cfg['version']}/{c['split']}/{c['clip_id']}.wav"
        sha = write_wav(split_dir / f"{c['clip_id']}.wav", q, sr)
        rows.append({"clip_id": c["clip_id"], "kind": c["kind"], "split": c["split"], "owner_base_id": c["owner"],
                     "variant_id": c["variant_id"], "reply_key": c.get("reply_key"), "used_by_probes": c["used_by"],
                     "text": c["text"], "snr_nominal_db": c["snr"], "replicate": c["rep"], "clean_clip_id": c["clean_id"],
                     "path": rel, "sha256": sha, "sample_rate": sr, "channels": 1, "subtype": "PCM_16",
                     **describe(q, cfg), **extra, "synthetic": True, "speaker": "tts:piper/en_US-lessac-medium"})
    rows.sort(key=lambda r: r["clip_id"])
    (partial / "audio_manifest.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    man = {"version": cfg["version"], "synthetic": True,
           "note": "All speech is TTS; it does not represent natural human speech. Annotations are unreviewed AI drafts.",
           "phase1_snapshot": cfg["phase1_snapshot"],
           "phase1_dataset_sha256": sha_file(ROOT / cfg["phase1_snapshot"] / "base_items.jsonl"),
           "config_sha256": sha_file(Path(args.config)), "snr_levels": cfg["snr_levels"], "master_seed": cfg["master_seed"],
           "counts": {"clips": len(rows), "initial": sum(r["kind"] == "initial" for r in rows),
                      "reply": sum(r["kind"] == "reply" for r in rows),
                      "clean_syntheses": sum(r["snr_nominal_db"] is None for r in rows),
                      "by_split": {s: sum(r["split"] == s for r in rows) for s in ("train", "val", "test")}},
           "versions": versions(), "noise_asset": cfg["noise"]["asset_id"],
           "audio_manifest_sha256": sha_file(partial / "audio_manifest.jsonl"),
           "code_sha256": {f: sha_file(ROOT / f) for f in ("scripts/generate_phase2_audio.py", "src/noise.py", "src/tts.py")}}
    (partial / "manifest.json").write_text(json.dumps(man, indent=2, sort_keys=True) + "\n")
    (partial / "run_log.json").write_text(json.dumps({"seconds": round(time.time() - t0, 1),
                                                    "note": "timing kept out of manifest.json to preserve reproducibility"}, indent=2) + "\n")
    partial.rename(final)
    print(f"built {len(rows)} clips -> {final} in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
