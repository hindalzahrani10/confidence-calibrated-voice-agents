"""When-to-clarify policies.

Each policy is a function from the confidence signals of one utterance to a
boolean: should the agent spend a turn asking a clarifying question?

The two trivial policies are not filler. `never_clarify` is the lower bound
on user burden and `always_clarify` is the upper bound on both burden and
achievable success, and the whole paper is the claim that a learned policy
sits above the line joining them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from .signals import SIGNAL_NAMES, ConfidenceSignals

Policy = Callable[[ConfidenceSignals], bool]


def never_clarify(_signals: ConfidenceSignals) -> bool:
    """Answer-only baseline: the burden floor."""
    return False


def always_clarify(_signals: ConfidenceSignals) -> bool:
    """Ask every time: the burden ceiling."""
    return True


def threshold_policy(signal_name: str, threshold: float) -> Policy:
    """Clarify when one signal falls below a threshold.

    The threshold MUST be tuned on the validation split only. Tuning it on
    test invalidates every headline number, and it is the single easiest
    mistake to make in this project.
    """
    if signal_name not in SIGNAL_NAMES:
        raise ValueError(f"unknown signal {signal_name!r}; expected one of {SIGNAL_NAMES}")

    def policy(signals: ConfidenceSignals) -> bool:
        return getattr(signals, signal_name) < threshold

    policy.__name__ = f"{signal_name}_threshold"
    return policy


def self_interrogation_policy(signals: ConfidenceSignals) -> bool:
    """Trust the model's own yes/no judgement about needing clarification."""
    return signals.self_interrogation < 0.5


@dataclass
class CalibratedFusion:
    """The proposed policy: logistic fusion over all five signals.

    Trained to predict whether clarification would *change the outcome*,
    rather than whether the answer is merely uncertain. That distinction is
    the value-of-information framing: an agent that is unsure but would get
    the same answer either way should not spend the user's turn.
    """

    model: Optional[object] = None
    threshold: float = 0.5

    def fit(
        self,
        signals: Sequence[ConfidenceSignals],
        clarification_helps: Sequence[bool],
    ) -> "CalibratedFusion":
        """Fit on the validation split.

        `clarification_helps[i]` is True when the agent answered wrongly
        without clarification but correctly with it: exactly the cases
        where asking bought something.
        """
        from sklearn.linear_model import LogisticRegression

        X = np.stack([s.as_vector() for s in signals])
        y = np.asarray(clarification_helps, dtype=int)

        if len(np.unique(y)) < 2:
            # Degenerate validation split: fall back to never clarifying
            # rather than fitting a model that cannot discriminate.
            self.model = None
            return self

        self.model = LogisticRegression(max_iter=1000, class_weight="balanced")
        self.model.fit(X, y)
        return self

    def probability(self, signals: ConfidenceSignals) -> float:
        """P(clarification would change the outcome)."""
        if self.model is None:
            return 0.0
        return float(self.model.predict_proba(signals.as_vector().reshape(1, -1))[0, 1])

    def __call__(self, signals: ConfidenceSignals) -> bool:
        return self.probability(signals) >= self.threshold

    def coefficients(self) -> Dict[str, float]:
        """Per-signal weights, for the analysis section."""
        if self.model is None:
            return {name: 0.0 for name in SIGNAL_NAMES}
        return dict(zip(SIGNAL_NAMES, self.model.coef_[0].tolist()))


def build_policies(cfg) -> Dict[str, Policy]:
    """Instantiate every policy named in the config's `policies.active` list.

    `calibrated_fusion` is returned unfitted; `run_agent.py` fits it on the
    validation split before evaluating on test.
    """
    thresholds = cfg["policies"]["thresholds"]
    available: Dict[str, Policy] = {
        "never_clarify": never_clarify,
        "always_clarify": always_clarify,
        "asr_threshold": threshold_policy(
            "asr_confidence", thresholds["asr_threshold"]
        ),
        "logprob_threshold": threshold_policy(
            "token_logprob",
            # Config stores the raw log-probability; signals are normalised.
            max(0.0, min(1.0, (thresholds["logprob_threshold"] + 3.0) / 3.0)),
        ),
        "verbalized_threshold": threshold_policy(
            "verbalized", thresholds["verbalized_threshold"] / 100.0
        ),
        "entropy_threshold": threshold_policy(
            "semantic_entropy", thresholds["entropy_threshold"]
        ),
        "self_interrogation": self_interrogation_policy,
        "calibrated_fusion": CalibratedFusion(),
    }

    active: List[str] = cfg["policies"]["active"]
    unknown = set(active) - set(available)
    if unknown:
        raise ValueError(f"unknown policies in config: {sorted(unknown)}")

    return {name: available[name] for name in active}
