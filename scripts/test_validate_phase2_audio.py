"""Negative tests for scripts/validate_phase2_audio.py.

Each case corrupts a TEMPORARY copy of the audio dataset in one way and
asserts the validator rejects it for the right reason. The real dataset is
only read.

    TMPDIR=<scratch> python scripts/test_validate_phase2_audio.py
"""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import validate_phase2_audio as V  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data/phase2/audio.v1"
CONFIG = ROOT / "configs/phase2_audio.yaml"


class Ctx:
    def __init__(self, tmp: Path):
        self.root = tmp / "audio.v1"
        shutil.copytree(SRC, self.root)
        self.rows = [json.loads(l) for l in (self.root / "audio_manifest.jsonl").read_text().splitlines()]
        self.man = json.loads((self.root / "manifest.json").read_text())

    def row(self, **where):
        return next(r for r in self.rows if all(r.get(k) == v for k, v in where.items()))

    def wav(self, r):
        return self.root / r["split"] / f"{r['clip_id']}.wav"

    def rewrite(self, r, q, sr=16000):
        sf.write(self.wav(r), q, sr, subtype="PCM_16")
        r["sha256"] = hashlib.sha256(self.wav(r).read_bytes()).hexdigest()

    def save(self, fix_hash=True):
        p = self.root / "audio_manifest.jsonl"
        p.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in self.rows))
        if fix_hash:
            self.man["audio_manifest_sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
        (self.root / "manifest.json").write_text(json.dumps(self.man, sort_keys=True))


def noisy(c): return c.row(kind="initial", snr_nominal_db=10, split="train")
def clean(c): return c.row(kind="initial", snr_nominal_db=None, split="train")


def raw_edit(c):  # bytes changed, manifest hash not updated
    p = c.wav(noisy(c)); b = bytearray(p.read_bytes()); b[100] ^= 0x7F; p.write_bytes(bytes(b)); c.save()

def rescaled_noisy(c):  # consistent hash, but no longer clean+gain*noise
    r = noisy(c); q, _ = sf.read(c.wav(r), dtype="int16"); c.rewrite(r, (q * 0.8).astype(np.int16)); c.save()

def wrong_nominal(c):
    noisy(c)["snr_nominal_db"] = 20; c.save()

def wrong_samplerate(c):
    r = clean(c); q, _ = sf.read(c.wav(r), dtype="int16"); c.rewrite(r, q, sr=22050); c.save()

def stereo(c):
    r = clean(c); q, _ = sf.read(c.wav(r), dtype="int16"); c.rewrite(r, np.stack([q, q], axis=1)); c.save()

def clipped(c):
    r = clean(c); q, _ = sf.read(c.wav(r), dtype="int16"); q[20000:20050] = 32767; c.rewrite(r, q); c.save()

def wrong_split_dir(c):
    r = clean(c); r["path"] = r["path"].replace("/train/", "/test/"); c.save()

def offset_outside_partition(c):
    noisy(c)["noise"]["offset_samples"] = 250 * 16000; c.save()

def missing_file(c):
    c.wav(noisy(c)).unlink()

def leaking_name(c):
    r = clean(c); r["clip_id"] = "p1_none_01@clean"; c.save()

def padding_not_silent(c):
    r = clean(c); q, _ = sf.read(c.wav(r), dtype="int16"); q[100:200] = 500; c.rewrite(r, q); c.save()

def not_synthetic(c):
    c.rows[0]["synthetic"] = False; c.save()

def manifest_hash(c):
    c.rows[0]["text"] += " "; c.save(fix_hash=False)

def drop_reply_noisy(c):
    r = next(x for x in c.rows if x["kind"] == "reply" and x["snr_nominal_db"] == 5)
    c.rows.remove(r); c.save()

def unnecessary_duplicate(c):
    a = next(x for x in c.rows if x["kind"] == "reply" and x["snr_nominal_db"] is None and x["owner_base_id"] == "p1_none_01")
    b = next(x for x in c.rows if x["kind"] == "reply" and x["snr_nominal_db"] is None and x["owner_base_id"] == "p1_none_01" and x is not a)
    shutil.copy(c.wav(a), c.wav(b)); b["sha256"] = a["sha256"]; c.save()

def clean_level_off(c):
    r = clean(c); q, _ = sf.read(c.wav(r), dtype="int16"); c.rewrite(r, (q * 0.5).astype(np.int16)); c.save()

def wrong_owner_split(c):
    r = clean(c); r["split"] = "val"; c.save()

CASES = [
    ("file bytes altered after hashing", raw_edit, "file hash"),
    ("noisy clip not rebuildable from clean+noise+gain", rescaled_noisy, "cannot be rebuilt"),
    ("nominal SNR wrong", wrong_nominal, "achieved SNR"),
    ("wrong sample rate", wrong_samplerate, "wrong format"),
    ("stereo file", stereo, "wrong format"),
    ("clipping", clipped, "clipping"),
    ("clip stored under the wrong split dir", wrong_split_dir, "wrong split directory"),
    ("noise slice outside the split partition", offset_outside_partition, "outside the"),
    ("clip file missing", missing_file, "file missing"),
    ("agent-visible-style id leaks a label", leaking_name, "not opaque"),
    ("padding is not digital silence", padding_not_silent, "padding"),
    ("synthetic flag removed", not_synthetic, "synthetic"),
    ("manifest hash tampered", manifest_hash, "hash"),
    ("reply audio missing at one SNR", drop_reply_noisy, "missing or incomplete"),
    ("unnecessary duplicate waveform", unnecessary_duplicate, "duplicate waveform"),
    ("clean level off target", clean_level_off, "off target"),
    ("clip split differs from Phase 1", wrong_owner_split, "split differs"),
]


def main() -> int:
    failures = 0
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        errs, _ = V.validate(SRC, CONFIG)
        print(f"{'ok  ' if not errs else 'FAIL'} real dataset passes ({len(errs)} errors)")
        failures += bool(errs)
        for name, fn, expect in CASES:
            c = Ctx(tmp)
            fn(c)
            errs, _ = V.validate(c.root, CONFIG)
            good = bool(errs) and any(expect in e for e in errs)
            failures += not good
            print(f"{'ok  ' if good else 'FAIL'} {name}" + ("" if good else f"   errors={errs[:2]}"))
            shutil.rmtree(c.root)
    n = len(CASES) + 1
    print(f"\n{n - failures}/{n} behaved as required")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
