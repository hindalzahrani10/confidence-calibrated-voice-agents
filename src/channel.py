"""Agent-side reply parsing, blind SNR estimation and the clarification-channel table.

Everything here sees only what a deployed agent sees: audio samples, ASR output,
and its own question. No gold labels, no dataset-private fields.

Channel model (notes/Formulation.md §4.1): p(ŷ | y, SNR, a_q) is estimated empirically from the
recognition rate of sample replies per SNR bucket and question type. Here
  p(ŷ = T_q(y) | y) = r_q(SNR)        (the reply is recognised as intended)
  p(ŷ = o      | y) = (1 - r_q)/(|O_q|-1)   for every other outcome o,
with r_q shrunk toward its SNR-pooled value (hierarchical Beta smoothing), so sparse cells
do not give extreme probabilities.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .ontology import STOP, match, tokens

YES = {"yes", "yeah", "yep", "yup", "correct", "right", "sure", "ok"}
NO = {"no", "nope", "nah", "wrong"}
NONE_MARK = {"none", "neither", "nothing"}
BUCKETS = ["clean", "20", "10", "5"]


# ---------------------------------------------------------------------------- parsers -------
def parse_confirm(text: str) -> Tuple[str, List[str]]:
    """('yes'|'no'|'unparseable', remaining tokens). The remainder of a 'no' may carry a correction."""
    toks = tokens(text)
    if toks and toks[0] in YES:
        return "yes", toks[1:]
    if toks and toks[0] in NO:
        return "no", toks[1:]
    return "unparseable", toks


def parse_closed(text: str, option_texts: Sequence[str], accept: float = 0.6) -> str:
    """Index (as str) of the option the reply selects, 'none', or 'unparseable'."""
    toks = [t for t in tokens(text) if t not in STOP]
    if any(t in NONE_MARK for t in tokens(text)):
        return "none"
    best, best_i = 0.0, -1
    for i, o in enumerate(option_texts):
        ot = [t for t in tokens(o) if t not in STOP]
        if not ot or not toks:
            continue
        cov_o = float(np.mean([max(match(c, t) for t in toks) for c in ot]))
        cov_t = float(np.mean([max(match(t, c) for c in ot) for t in toks]))
        sc = 0.5 * (cov_o + cov_t)
        if sc > best:
            best, best_i = sc, i
    return str(best_i) if best >= accept else "unparseable"


# --------------------------------------------------------------------- blind SNR estimate ----
def snr_feature(x: np.ndarray, sr: int = 16000) -> Optional[float]:
    """Statistic z (dB) from the audio alone: speech-frame power over the noise floor.

    The noise floor is measured in the recording's lead-in and tail, which are speech-free by construction;
    speech frames are those 6 dB above that floor. Returns None when the floor is digital silence (clean).
    """
    lead, tail = x[: int(0.4 * sr)], x[-int(0.5 * sr):]
    n = float(np.mean(np.concatenate([lead, tail]).astype(np.float64) ** 2))
    if n < 1e-9:
        return None
    frame, hop = 320, 160
    idx = np.arange(frame)[None, :] + hop * np.arange(1 + (len(x) - frame) // hop)[:, None]
    e = np.mean(x.astype(np.float64)[idx] ** 2, axis=1)
    act = e[e > 4.0 * n]
    return 10 * math.log10(max(float(act.mean()) if len(act) else n, n) / n)


def estimate_snr_bucket(x: np.ndarray, centroids: Dict[str, float], sr: int = 16000) -> str:
    """Nearest-centroid bucket on the audio statistic; centroids are fitted on TRAIN clips only."""
    z = snr_feature(x, sr)
    if z is None:
        return "clean"
    return min(centroids, key=lambda b: abs(z - centroids[b]))


# ------------------------------------------------------------------------ channel table -----
@dataclass
class ChannelTable:
    """r[(qtype, bucket)] with counts; `mode` selects how the agent may use it."""

    counts: Dict[str, Dict[str, List[int]]]  # qtype -> bucket -> [successes, trials]
    prior_strength: float = 4.0

    def pooled(self, q: str) -> Tuple[int, int]:
        k = sum(v[0] for v in self.counts[q].values())
        n = sum(v[1] for v in self.counts[q].values())
        return k, n

    def r(self, q: str, bucket: str, mode: str = "snr") -> float:
        """mode: 'snr' (NM-VoI), 'pooled' (noise-independent), 'perfect' (standard EVPI)."""
        if mode == "perfect":
            return 1.0 - 1e-6
        k, n = self.pooled(q)
        r_pool = (k + 1.0) / (n + 2.0)
        if mode == "pooled":
            return r_pool
        kb, nb = self.counts[q].get(bucket, [0, 0])
        m = self.prior_strength
        return (kb + m * r_pool) / (nb + m)

    def to_json(self) -> dict:
        return {"counts": self.counts, "prior_strength": self.prior_strength}

    @classmethod
    def from_json(cls, d: dict) -> "ChannelTable":
        return cls(counts=d["counts"], prior_strength=d.get("prior_strength", 4.0))


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))
