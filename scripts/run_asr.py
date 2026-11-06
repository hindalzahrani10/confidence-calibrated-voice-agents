"""Run Whisper ASR over every Phase 2 clip. Resumable and append-only.

Output rows hold ONLY what an agent could observe from a recording (transcript,
word strings + probabilities, decoder scores) plus runtime/config. They are keyed
by opaque clip_id; gold transcripts, splits and conditions are NOT written here
and are joined in later, evaluation-side, from the private audio manifest.

    python scripts/run_asr.py [--audio data/phase2/audio.v1] [--out data/phase3/asr.v1]

Deviations from faster-whisper defaults (recorded in config.json): temperature=0.0
(no random fallback sampling, so a re-run is deterministic) and int8 on CPU.
The existing src/asr.py wrapper is not used because it discards the word strings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audio", default=str(ROOT / "data/phase2/audio.v1"))
    ap.add_argument("--out", default=str(ROOT / "data/phase3/asr.v1"))
    ap.add_argument("--limit", type=int, default=0, help="debug: stop after N new clips")
    a = ap.parse_args()
    audio, out = Path(a.audio), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load((ROOT / "configs/default.yaml").read_text())["asr"]
    model_dir = ROOT / "data/models/faster-whisper-small"
    settings = {"model": "faster-whisper-small (Systran/faster-whisper-small)", "model_bin_sha256":
                hashlib.sha256((model_dir / "model.bin").read_bytes()).hexdigest(), "language": cfg["language"],
                "beam_size": cfg["beam_size"], "word_timestamps": True, "temperature": 0.0, "vad_filter": False,
                "device": "cpu", "compute_type": "int8", "cpu_threads": 2,
                "audio_manifest_sha256": hashlib.sha256((audio / "audio_manifest.jsonl").read_bytes()).hexdigest()}
    cfg_path = out / "config.json"
    if cfg_path.exists():
        old = json.loads(cfg_path.read_text())
        if {k: v for k, v in old.items() if k != "faster_whisper"} != {k: v for k, v in settings.items() if k != "faster_whisper"}:
            raise SystemExit("existing ASR output was made with different settings; use a new --out version")
    import importlib.metadata as md
    settings["faster_whisper"] = md.version("faster-whisper")
    settings["ctranslate2"] = md.version("ctranslate2")
    cfg_path.write_text(json.dumps(settings, indent=2, sort_keys=True) + "\n")

    rows = [json.loads(l) for l in (audio / "audio_manifest.jsonl").read_text().splitlines()]
    order = {"initial": 0, "reply": 1}
    rows.sort(key=lambda r: (order[r["kind"]], {"train": 0, "val": 1, "test": 2}[r["split"]], r["clip_id"]))
    res_path = out / "asr_results.jsonl"
    done = set()
    if res_path.exists():
        done = {json.loads(l)["clip_id"] for l in res_path.read_text().splitlines() if l.strip()}
    todo = [r for r in rows if r["clip_id"] not in done]
    print(f"{len(done)} done, {len(todo)} to do", flush=True)

    from faster_whisper import WhisperModel
    model = WhisperModel(str(model_dir), device="cpu", compute_type="int8", cpu_threads=2)
    n_new = 0
    with open(res_path, "a") as fh:
        for r in todo:
            x, sr = sf.read(audio / r["split"] / f"{r['clip_id']}.wav", dtype="float32")
            t0 = time.time()
            segs, info = model.transcribe(x, language=cfg["language"], beam_size=cfg["beam_size"],
                                          word_timestamps=True, temperature=0.0)
            segs = list(segs)
            dt = time.time() - t0
            words = [{"w": w.word.strip(), "p": round(float(w.probability), 5), "t0": round(float(w.start), 3),
                      "t1": round(float(w.end), 3)} for s in segs for w in (s.words or [])]
            rec = {"clip_id": r["clip_id"], "text": "".join(s.text for s in segs).strip(), "words": words,
                   "avg_logprob": [round(float(s.avg_logprob), 5) for s in segs],
                   "no_speech_prob": [round(float(s.no_speech_prob), 5) for s in segs],
                   "compression_ratio": [round(float(s.compression_ratio), 4) for s in segs],
                   "language_prob": round(float(info.language_probability), 5), "runtime_s": round(dt, 3)}
            fh.write(json.dumps(rec, sort_keys=True) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
            n_new += 1
            if n_new % 25 == 0:
                print(f"{len(done) + n_new}/{len(rows)}", flush=True)
            if a.limit and n_new >= a.limit:
                break
    print("finished" if not a.limit else "limit reached", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
