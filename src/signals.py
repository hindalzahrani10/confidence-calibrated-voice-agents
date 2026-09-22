"""The confidence signal bank: the heart of the project.

Five families are extracted per utterance and logged side by side, so that
the calibration analysis can ask a fair question: which of these actually
predicts failure, and does the answer change when the audio gets noisy?

Every signal is normalised to [0, 1] where HIGHER MEANS MORE CONFIDENT.
Keeping that convention uniform is what lets policies and metrics treat the
five families interchangeably.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Sequence

import numpy as np

# The canonical signal names, in the order they appear in the paper tables.
SIGNAL_NAMES = (
    "asr_confidence",
    "token_logprob",
    "verbalized",
    "semantic_entropy",
    "self_interrogation",
)


@dataclass
class ConfidenceSignals:
    """All five signal families for one utterance, each in [0, 1]."""

    asr_confidence: float = 0.0
    token_logprob: float = 0.0
    verbalized: float = 0.5
    semantic_entropy: float = 0.0
    self_interrogation: float = 0.5

    # Diagnostics kept alongside the normalised values, for error analysis.
    asr_min_word: float = 0.0
    asr_nbest_disagreement: float = 0.0
    raw_mean_logprob: float = 0.0
    n_semantic_clusters: int = 1

    def as_vector(self) -> np.ndarray:
        """The five signals as a feature vector for the fusion policy."""
        return np.array([getattr(self, name) for name in SIGNAL_NAMES], dtype=float)

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def normalise_logprob(mean_logprob: float, floor: float = -3.0) -> float:
    """Map a mean token log-probability onto [0, 1].

    `floor` is the log-probability treated as zero confidence. -3.0 is a
    reasonable default for chat models; check the histogram in your EDA
    notebook and adjust if your model lives in a different range.
    """
    if floor >= 0.0:
        raise ValueError("floor must be negative")
    scaled = (mean_logprob - floor) / (0.0 - floor)
    return float(np.clip(scaled, 0.0, 1.0))


def cluster_by_meaning(answers: Sequence[str]) -> List[List[int]]:
    """Group answers into meaning-equivalent clusters.

    This is the cheap stand-in for the bidirectional-entailment clustering
    used in the semantic-entropy literature: normalise and compare token
    sets. Replacing it with a real NLI model is the first upgrade worth
    making, and it is a clean ablation for the paper.
    """
    clusters: List[List[int]] = []
    keys: List[set] = []

    for i, ans in enumerate(answers):
        tokens = {t.strip(".,!?;:").lower() for t in ans.split() if t.strip(".,!?;:")}
        placed = False
        for j, key in enumerate(keys):
            union = tokens | key
            if union and len(tokens & key) / len(union) >= 0.6:
                clusters[j].append(i)
                keys[j] = key | tokens
                placed = True
                break
        if not placed:
            clusters.append([i])
            keys.append(tokens)

    return clusters


def semantic_entropy(answers: Sequence[str]) -> tuple[float, int]:
    """Entropy over meaning clusters of k sampled answers.

    Returns (normalised_entropy, n_clusters) where the entropy is divided
    by log(k) so it lands in [0, 1]: 0 means every sample meant the same
    thing, 1 means every sample meant something different.
    """
    if len(answers) <= 1:
        return 0.0, 1

    clusters = cluster_by_meaning(answers)
    counts = np.array([len(c) for c in clusters], dtype=float)
    probs = counts / counts.sum()
    entropy = float(-np.sum(probs * np.log(probs)))
    max_entropy = float(np.log(len(answers)))

    normalised = entropy / max_entropy if max_entropy > 0 else 0.0
    return float(np.clip(normalised, 0.0, 1.0)), len(clusters)


def build_signals(
    asr_result,
    answer_gen,
    sampled_answers: Sequence[str],
    verbalized: float,
    self_clarify_flag: bool,
    nbest_disagreement: float = 0.0,
) -> ConfidenceSignals:
    """Assemble all five signals for one utterance.

    Note the two inversions: semantic entropy and the self-interrogation
    flag are *uncertainty* measures, so they are flipped to keep the
    higher-is-more-confident convention.
    """
    entropy, n_clusters = semantic_entropy(sampled_answers)

    return ConfidenceSignals(
        asr_confidence=float(asr_result.mean_word_confidence),
        token_logprob=normalise_logprob(answer_gen.mean_logprob),
        verbalized=float(verbalized),
        semantic_entropy=1.0 - entropy,
        self_interrogation=0.0 if self_clarify_flag else 1.0,
        asr_min_word=float(asr_result.min_word_confidence),
        asr_nbest_disagreement=float(nbest_disagreement),
        raw_mean_logprob=float(answer_gen.mean_logprob),
        n_semantic_clusters=int(n_clusters),
    )


def attribute_uncertainty_source(signals: ConfidenceSignals) -> str:
    """Guess *why* the agent is uncertain, from the shape of the signals.

    The intuition the project tests: low ASR confidence with coherent
    samples means the agent misheard; high ASR confidence with scattered
    samples means the request itself is ambiguous.

    This rule-based version is the baseline. Metric 9 in the roadmap
    (uncertainty-source attribution accuracy) scores it against the
    `ambiguity_type` labels, and beating it with a learned classifier is a
    natural extension.
    """
    acoustic_doubt = signals.asr_confidence < 0.7
    semantic_doubt = signals.semantic_entropy < 0.6

    if acoustic_doubt and not semantic_doubt:
        return "acoustic"
    if semantic_doubt and not acoustic_doubt:
        return "intent"
    if acoustic_doubt and semantic_doubt:
        # Both channels unhappy: in a voice agent the acoustic problem is
        # the one worth repairing first, since it gates everything after.
        return "acoustic"
    return "none"
