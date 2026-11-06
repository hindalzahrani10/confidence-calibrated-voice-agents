"""Fill logged on-demand reply requests: TTS + noise (same code and seeds as Phase 2) + ASR (same settings as run_asr).

Run only AFTER scripts/run_asr.py has finished (two ASR processes would fight over 2 vCPUs and 3.8 GB RAM).
Outputs (versioned, append-only): data/phase2/ondemand.v1/ (audio + manifest), data/phase3/asr_ondemand.v1/.
"""
import hashlib, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np, soundfile as sf, yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import generate_phase2_audio as G  # noqa: E402

if subprocess.run(["pgrep", "-f", "scripts/run_asr.py"], capture_output=True).stdout.strip():
    raise SystemExit("scripts/run_asr.py is still running; wait for it to finish")
req_path = ROOT / "data/phase3/ondemand_requests.jsonl"
reqs = [json.loads(l) for l in req_path.read_text().splitlines() if l.strip()] if req_path.exists() else []
out_audio, out_asr = ROOT / "data/phase2/ondemand.v1", ROOT / "data/phase3/asr_ondemand.v1"
out_audio.mkdir(parents=True, exist_ok=True); out_asr.mkdir(parents=True, exist_ok=True)
man_path, asr_path = out_audio / "ondemand_manifest.jsonl", out_asr / "asr_results.jsonl"
done = {json.loads(l)["clip_id"] for l in asr_path.read_text().splitlines() if l.strip()} if asr_path.exists() else set()
todo = [r for r in reqs if r["clip_id"] not in done]
print(f"{len(reqs)} requested, {len(todo)} to generate", flush=True)
if not todo:
    raise SystemExit(0)
cfg = G.load_cfg(ROOT / "configs/phase2_audio.yaml")
B = G.Builder(cfg); sr = cfg["format"]["sample_rate"]
asr_cfg = json.loads((ROOT / "data/phase3/asr.v1/config.json").read_text())
from faster_whisper import WhisperModel
model = WhisperModel(str(ROOT / "data/models/faster-whisper-small"), device="cpu", compute_type="int8", cpu_threads=2)
clean_cache = {}
with open(man_path, "a") as mf, open(asr_path, "a") as af:
    for r in todo:
        key = (r["base_id"], r["text"].lower())
        if key not in clean_cache:
            clean_cache[key] = B.clean(r["text"])[0]
        cq = clean_cache[key]
        if r["snr"] is None:
            q, info = cq, {}
        else:
            q, info = B.noisy(cq, r["split"], r["clip_id"], r["snr"])
        p = out_audio / r["split"] / f"{r['clip_id']}.wav"
        sha = G.write_wav(p, q, sr)
        x, _ = sf.read(p, dtype="float32")
        t0 = time.time()
        segs, inf = model.transcribe(x, language=asr_cfg["language"], beam_size=asr_cfg["beam_size"], word_timestamps=True, temperature=0.0)
        segs = list(segs); dt = time.time() - t0
        words = [{"w": w.word.strip(), "p": round(float(w.probability), 5), "t0": round(float(w.start), 3), "t1": round(float(w.end), 3)}
                 for s in segs for w in (s.words or [])]
        af.write(json.dumps({"clip_id": r["clip_id"], "text": "".join(s.text for s in segs).strip(), "words": words,
                             "avg_logprob": [round(float(s.avg_logprob), 5) for s in segs],
                             "no_speech_prob": [round(float(s.no_speech_prob), 5) for s in segs],
                             "compression_ratio": [round(float(s.compression_ratio), 4) for s in segs],
                             "language_prob": round(float(inf.language_probability), 5), "runtime_s": round(dt, 3)}, sort_keys=True) + "\n")
        af.flush(); os.fsync(af.fileno())
        mf.write(json.dumps({**r, "path": str(p.relative_to(ROOT)), "sha256": sha, "synthetic": True,
                             "measured_snr_db_from_files": info.get("measured_snr_db_from_files")}, sort_keys=True) + "\n")
        mf.flush()
print("done")
