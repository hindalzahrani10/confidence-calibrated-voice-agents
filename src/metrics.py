"""The twelve project metrics.

Split into two groups:

  * calibration metrics  : how well does a confidence signal track
                           correctness? (ECE, Brier, AUROC, risk-coverage)
  * behaviour metrics    : what does the clarification policy actually do?
                           (precision, recall, over-clarification, success)

The rule the whole paper depends on: user burden is ALWAYS reported next to
accuracy. Always-clarify wins on accuracy alone, which is precisely why
accuracy alone is the wrong headline.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------

def expected_calibration_error(
    confidences: Sequence[float], correct: Sequence[bool], n_bins: int = 15
) -> float:
    """Standard binned ECE.

    Known pitfall: ECE is sensitive to the number of bins, which is why we
    report Brier score next to it rather than in place of it.
    """
    conf = np.asarray(confidences, dtype=float)
    acc = np.asarray(correct, dtype=float)
    if conf.size == 0:
        return float("nan")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        # Include the left edge on the first bin so conf == 0 is counted.
        mask = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if not mask.any():
            continue
        weight = mask.mean()
        ece += weight * abs(acc[mask].mean() - conf[mask].mean())
    return float(ece)


def brier_score(confidences: Sequence[float], correct: Sequence[bool]) -> float:
    """Proper scoring rule: mean squared error of the confidence estimate."""
    conf = np.asarray(confidences, dtype=float)
    acc = np.asarray(correct, dtype=float)
    if conf.size == 0:
        return float("nan")
    return float(np.mean((conf - acc) ** 2))


def auroc(confidences: Sequence[float], correct: Sequence[bool]) -> float:
    """Does the signal rank correct answers above wrong ones?

    Distinct from calibration: a signal can rank perfectly (AUROC 1.0) yet
    be badly calibrated, and a constant signal is perfectly calibrated on
    average yet useless for ranking. Reporting both is the point.
    """
    from sklearn.metrics import roc_auc_score

    acc = np.asarray(correct, dtype=int)
    if len(np.unique(acc)) < 2:
        return float("nan")
    return float(roc_auc_score(acc, np.asarray(confidences, dtype=float)))


def risk_coverage_curve(
    confidences: Sequence[float], correct: Sequence[bool]
) -> Tuple[np.ndarray, np.ndarray]:
    """Risk (error rate) as a function of coverage, answering most-confident first."""
    conf = np.asarray(confidences, dtype=float)
    acc = np.asarray(correct, dtype=float)
    if conf.size == 0:
        return np.array([]), np.array([])

    order = np.argsort(-conf)
    sorted_acc = acc[order]
    n = len(sorted_acc)

    coverage = np.arange(1, n + 1) / n
    risk = 1.0 - np.cumsum(sorted_acc) / np.arange(1, n + 1)
    return coverage, risk


def risk_coverage_auc(confidences: Sequence[float], correct: Sequence[bool]) -> float:
    """Area under the risk-coverage curve; lower is better."""
    coverage, risk = risk_coverage_curve(confidences, correct)
    if coverage.size == 0:
        return float("nan")
    return float(np.trapezoid(risk, coverage))


def temperature_scale(
    confidences: Sequence[float], correct: Sequence[bool]
) -> float:
    """Fit a single temperature on the validation split.

    Returns the temperature T that minimises ECE when applied as
    `conf ** T`. Deliberately a 1-parameter fit: with a few hundred
    validation items, anything richer will overfit.
    """
    conf = np.clip(np.asarray(confidences, dtype=float), 1e-6, 1.0)
    best_t, best_ece = 1.0, float("inf")
    for t in np.linspace(0.2, 5.0, 49):
        ece = expected_calibration_error(conf**t, correct)
        if ece < best_ece:
            best_t, best_ece = float(t), ece
    return best_t


# --------------------------------------------------------------------------
# Clarification behaviour
# --------------------------------------------------------------------------

def clarification_metrics(
    asked: Sequence[bool], needed: Sequence[bool]
) -> Dict[str, float]:
    """Precision, recall and over-clarification rate for a policy.

    `needed[i]` is True when the item genuinely required clarification
    (its ambiguity type is not "none").
    """
    a = np.asarray(asked, dtype=bool)
    n = np.asarray(needed, dtype=bool)

    true_pos = float(np.sum(a & n))
    precision = true_pos / float(a.sum()) if a.sum() > 0 else float("nan")
    recall = true_pos / float(n.sum()) if n.sum() > 0 else float("nan")

    # User burden: of the items that needed nothing, how many were
    # interrupted anyway?
    clean = ~n
    over = float(np.sum(a & clean)) / float(clean.sum()) if clean.sum() > 0 else 0.0

    return {
        "clarification_precision": precision,
        "clarification_recall": recall,
        "over_clarification_rate": over,
        "ask_rate": float(a.mean()) if a.size else 0.0,
    }


def task_success_rate(correct: Sequence[bool]) -> float:
    """End-to-end correctness after the full interaction."""
    return float(np.mean(np.asarray(correct, dtype=float))) if len(correct) else 0.0


def attribution_accuracy(
    predicted: Sequence[str], gold: Sequence[str]
) -> float:
    """Metric 9: when uncertain, does the agent blame the right source?

    Scored only on items that genuinely carried uncertainty, since
    attributing a source to a clean item is not meaningful.
    """
    pairs = [(p, g) for p, g in zip(predicted, gold) if g != "none"]
    if not pairs:
        return float("nan")
    return float(np.mean([p == g for p, g in pairs]))


def calibration_drift(
    clean_ece: float, noisy_ece: float
) -> float:
    """Metric 10: how much calibration degrades when the audio does."""
    return float(noisy_ece - clean_ece)


# --------------------------------------------------------------------------
# Uncertainty on the metrics themselves
# --------------------------------------------------------------------------

def bootstrap_ci(
    values: Sequence[float],
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 1234,
) -> Tuple[float, float, float]:
    """Bootstrap a mean and its confidence interval over items.

    With a few hundred items, differences of a couple of points are noise.
    Every headline claim in the paper carries one of these.
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    means = np.array(
        [
            rng.choice(arr, size=arr.size, replace=True).mean()
            for _ in range(n_bootstrap)
        ]
    )
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(arr.mean()), float(lo), float(hi)


def summarise_signal(
    confidences: Sequence[float], correct: Sequence[bool], n_bins: int = 15
) -> Dict[str, float]:
    """Every calibration metric for one signal, as one row of the results table."""
    return {
        "ece": expected_calibration_error(confidences, correct, n_bins=n_bins),
        "brier": brier_score(confidences, correct),
        "auroc": auroc(confidences, correct),
        "rc_auc": risk_coverage_auc(confidences, correct),
        "n": len(confidences),
    }
