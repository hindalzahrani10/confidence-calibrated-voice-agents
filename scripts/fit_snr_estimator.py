"""Fit the agent's blind SNR-bucket rule: nearest centroid on the audio statistic, TRAIN clips only.

The nominal SNR label supervises the fit (development data); VAL is only used to report accuracy.
The agent at run time sees audio samples and these centroids, never a label.
"""
import json, sys
from collections import Counter
from pathlib import Path
import numpy as np, soundfile as sf
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.channel import estimate_snr_bucket, snr_feature  # noqa: E402

man = [json.loads(l) for l in (ROOT / "data/phase2/audio.v1/audio_manifest.jsonl").read_text().splitlines()]
feat = {}
for m in man:
    if m["split"] == "test":
        continue                      # sealed: not even the SNR statistic of test audio is examined
    x, _ = sf.read(ROOT / m["path"], dtype="float32")
    feat[m["clip_id"]] = (snr_feature(x), "clean" if m["snr_nominal_db"] is None else str(m["snr_nominal_db"]), m["split"], m["kind"], x)
cent = {}
for b in ("20", "10", "5"):
    z = [v[0] for v in feat.values() if v[2] == "train" and v[1] == b and v[0] is not None]
    cent[b] = float(np.mean(z))
print("centroids (train):", {k: round(v, 2) for k, v in cent.items()})
res = {}
for sp in ("train", "val"):
    conf = Counter()
    for z, true, s, kind, x in feat.values():
        if s == sp:
            conf[(true, estimate_snr_bucket(x, cent))] += 1
    tot = sum(conf.values()); ok = sum(v for (t, e), v in conf.items() if t == e)
    res[sp] = {"accuracy": ok / tot, "confusion": {f"{t}->{e}": v for (t, e), v in sorted(conf.items())}}
    print(sp, "accuracy", round(ok / tot, 3), dict(sorted(conf.items())))
out = ROOT / "data/phase4"; out.mkdir(exist_ok=True)
(out / "snr_estimator.v1.json").write_text(json.dumps({"centroids": cent, "fit_on": "train (initial + reply clips)", "results": res}, indent=1) + "\n")
