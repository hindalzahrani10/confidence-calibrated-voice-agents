"""EVALUATOR-SIDE metrics (these may use gold labels; agent code must never import this module)."""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import numpy as np

from .ontology import Task, tokens


def edit_distance(a: Sequence, b: Sequence) -> int:
    prev = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1]))
        prev = cur
    return prev[-1]


def wer(ref: str, hyp: str) -> Tuple[int, int]:
    """(word errors, reference words) after the shared number/time normalisation."""
    r, h = tokens(ref), tokens(hyp)
    return edit_distance(r, h), max(len(r), 1)


def cer(ref: str, hyp: str) -> Tuple[int, int]:
    r, h = " ".join(tokens(ref)), " ".join(tokens(hyp))
    return edit_distance(r, h), max(len(r), 1)


def accepted(task: Task, patterns: Sequence[dict]) -> bool:
    for pat in patterns:
        if task.intent == pat["intent"] and set(task.sdict) == set(pat["slots"]) and all(
                v == "*" or task.sdict[k] == v for k, v in pat["slots"].items()):
            return True
    return False


def accepted_mask(cat: Sequence[Task], patterns: Sequence[dict]) -> np.ndarray:
    return np.array([accepted(t, patterns) for t in cat])


def ece(conf: np.ndarray, correct: np.ndarray, bins: int = 15) -> float:
    if len(conf) == 0:
        return float("nan")
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(total)


def reliability(conf: np.ndarray, correct: np.ndarray, bins: int = 10):
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.any():
            out.append((float(conf[m].mean()), float(correct[m].mean()), int(m.sum())))
    return out


def brier(conf: np.ndarray, correct: np.ndarray) -> float:
    return float(np.mean((conf - correct) ** 2)) if len(conf) else float("nan")


def nll(mass: np.ndarray) -> float:
    return float(-np.mean(np.log(np.maximum(mass, 1e-6)))) if len(mass) else float("nan")


def cluster_bootstrap(values_by_cluster: Sequence[np.ndarray], n_boot: int = 2000, seed: int = 0,
                      stat=np.mean) -> Tuple[float, float, float]:
    """Mean and 95% CI resampling whole base-item clusters (the SNR variants of one item are not independent)."""
    rng = np.random.default_rng(seed)
    k = len(values_by_cluster)
    pt = float(stat(np.concatenate(values_by_cluster)))
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, k, k)
        boots.append(float(stat(np.concatenate([values_by_cluster[i] for i in idx]))))
    return pt, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
