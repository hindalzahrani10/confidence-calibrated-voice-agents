"""NM-VoI decision machinery and baselines (agent-side: sees only observations).

Implements notes/Formulation.md:
  U(a, y) = R(a, y) - C(a);   R = +1 correct answer, -1 wrong answer, 0 abstain.
  EU(answer) = E_{y~b0}[R]  =  2*max_y b0(y) - 1    (answer the MAP task)
  EU(ask a_q) = sum_o p(o) * max(0, 2*max_y b1(y|o) - 1) - C(a_q),
      p(o | y) = channel(o | T_q(y), SNR, q),   b1(y|o) ∝ b0(y) p(o|y).
One-step lookahead; after the reply the agent answers its MAP task or abstains.

Policies differ ONLY in what they assume about the reply channel (or which actions they
may take); all other computation is shared, so ablations change one component at a time:
  nm_voi        channel r_q(SNR bucket) estimated on dev data      (the proposal)
  noise_indep   channel r_q pooled over SNR                        (ablation: remove noise dependence)
  evpi          perfect channel r_q = 1                            (standard EVPI)
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .channel import ChannelTable
from .ontology import Task


@dataclass(frozen=True)
class Costs:
    open: float = 0.30
    closed: float = 0.20
    confirm: float = 0.10

    def of(self, qtype: str) -> float:
        return getattr(self, qtype)


@dataclass(frozen=True)
class Question:
    qtype: str                                  # 'open' | 'closed' | 'confirm'
    variable: Optional[str] = None              # closed: 'intent' or a slot name
    option_values: Tuple = ()                   # closed: canonical values offered
    option_texts: Tuple[str, ...] = ()
    proposal: Optional[int] = None              # confirm: catalogue index proposed


@dataclass
class Decision:
    kind: str                                   # 'answer' | 'abstain' | 'ask'
    question: Optional[Question] = None
    eus: Dict[str, float] = field(default_factory=dict)


# ------------------------------------------------------------------------ belief helpers ----
def entropy(b: np.ndarray) -> float:
    p = b[b > 0]
    return float(-(p * np.log(p)).sum())


def eu_answer(b: np.ndarray) -> float:
    return 2.0 * float(b.max()) - 1.0


def value_of(task: Task, var: str):
    return task.intent if var == "intent" else task.sdict.get(var)


def variables(cat: Sequence[Task]) -> List[str]:
    names = {"intent"}
    for t in cat:
        names.update(n for n, _ in t.slots)
    return sorted(names)


def marginal(cat: Sequence[Task], b: np.ndarray, var: str) -> Dict[object, float]:
    out: Dict[object, float] = {}
    for t, p in zip(cat, b):
        out[value_of(t, var)] = out.get(value_of(t, var), 0.0) + float(p)
    return out


def var_entropy(cat, b, var) -> float:
    m = marginal(cat, b, var)
    return entropy(np.array(list(m.values())))


def option_text(var: str, value) -> str:
    from .ontology import display, intent_phrase
    return display(var, value) if var != "intent" else intent_phrase(str(value))


def closed_question(cat, b, var: str, k: int = 3) -> Question:
    m = {v: p for v, p in marginal(cat, b, var).items() if v is not None}
    top = sorted(m, key=lambda v: (-m[v], str(v)))[:k]
    return Question("closed", var, tuple(top), tuple(option_text(var, v) for v in top))


# ------------------------------------------------------------------------- outcome models ---
def outcome_model(cat, b, q: Question, ch: ChannelTable, bucket: str, mode: str, m_support: int = 8):
    """Return (P[y, o], T[y]) with P[y,o] = p(observed outcome o | true task y)."""
    n = len(cat)
    if q.qtype == "confirm":
        r_yes, r_no = ch.r("confirm_yes", bucket, mode), ch.r("confirm_no", bucket, mode)
        P = np.zeros((n, 2))
        for y in range(n):
            p_yes = r_yes if y == q.proposal else 1.0 - r_no
            P[y] = [p_yes, 1.0 - p_yes]
        T = np.array([0 if y == q.proposal else 1 for y in range(n)])
        return P, T
    if q.qtype == "closed":
        r = ch.r("closed", bucket, mode)
        k = len(q.option_values)
        T = np.array([q.option_values.index(value_of(t, q.variable)) if value_of(t, q.variable) in q.option_values else k
                      for t in cat])
        n_out = k + 1
    else:  # open: the reply names the full task; support = the agent's m most plausible tasks
        r = ch.r("open", bucket, mode)
        support = list(np.argsort(-b)[:m_support])
        T = np.array([support.index(y) if y in support else len(support) for y in range(n)])
        n_out = len(support) + 1
    P = np.full((n, n_out), (1.0 - r) / max(n_out - 1, 1))
    P[np.arange(n), T] = r
    return P, T


def eu_ask(b, q: Question, cat, ch, bucket, mode, costs: Costs, allow_abstain: bool) -> float:
    P, _ = outcome_model(cat, b, q, ch, bucket, mode)
    joint = b[:, None] * P
    p_o = joint.sum(axis=0)
    total = 0.0
    for o in range(P.shape[1]):
        if p_o[o] <= 1e-12:
            continue
        v = eu_answer(joint[:, o] / p_o[o])
        total += p_o[o] * (max(v, 0.0) if allow_abstain else v)
    return total - costs.of(q.qtype)


def update(b: np.ndarray, P: np.ndarray, o: int) -> np.ndarray:
    post = b * P[:, o]
    s = post.sum()
    return post / s if s > 0 else b


# ----------------------------------------------------------------------------- policies -----
@dataclass
class AgentState:
    """Everything a policy may use: belief, the blind SNR bucket and the public catalogue."""

    cat: Sequence[Task]
    b: np.ndarray
    bucket: str


class Policy:
    name = "policy"
    allow_abstain = False

    def decide(self, s: AgentState) -> Decision:
        raise NotImplementedError

    def final(self, b: np.ndarray) -> str:
        return "abstain" if (self.allow_abstain and eu_answer(b) <= 0.0) else "answer"


class NeverAsk(Policy):
    name = "never_ask"

    def decide(self, s):
        return Decision("answer")


def _best_var(s: AgentState) -> str:
    return max(variables(s.cat), key=lambda v: (var_entropy(s.cat, s.b, v), v))


class AlwaysAsk(Policy):
    name = "always_ask"

    def decide(self, s):
        return Decision("ask", closed_question(s.cat, s.b, _best_var(s)))


class ConfidenceThreshold(Policy):
    """Ask a closed question when the top task's probability is below tau."""

    def __init__(self, tau: float):
        self.tau, self.name = tau, f"confidence_threshold"

    def decide(self, s):
        if float(s.b.max()) >= self.tau:
            return Decision("answer")
        return Decision("ask", closed_question(s.cat, s.b, _best_var(s)))


class EntropyThreshold(Policy):
    """Ask a closed question when belief entropy (nats) exceeds h."""

    def __init__(self, h: float):
        self.h, self.name = h, "entropy_threshold"

    def decide(self, s):
        if entropy(s.b) <= self.h:
            return Decision("answer")
        return Decision("ask", closed_question(s.cat, s.b, _best_var(s)))


class VoI(Policy):
    """One-step value-of-information policy; `mode` fixes the channel assumption."""

    def __init__(self, name: str, ch: ChannelTable, mode: str, costs: Costs,
                 allowed: Tuple[str, ...] = ("open", "closed", "confirm"), allow_abstain: bool = True,
                 n_closed_vars: int = 3):
        self.name, self.ch, self.mode, self.costs = name, ch, mode, costs
        self.allowed, self.allow_abstain, self.n_vars = allowed, allow_abstain, n_closed_vars

    def candidates(self, s: AgentState) -> List[Question]:
        qs: List[Question] = []
        if "confirm" in self.allowed:
            qs.append(Question("confirm", proposal=int(np.argmax(s.b))))
        if "open" in self.allowed:
            qs.append(Question("open"))
        if "closed" in self.allowed:
            top = sorted(variables(s.cat), key=lambda v: (-var_entropy(s.cat, s.b, v), v))[: self.n_vars]
            qs += [closed_question(s.cat, s.b, v) for v in top if var_entropy(s.cat, s.b, v) > 1e-9]
        return qs

    def decide(self, s):
        eus = {"answer": eu_answer(s.b)}
        if self.allow_abstain:
            eus["abstain"] = 0.0
        best_q, best = None, max(eus.values())
        for q in self.candidates(s):
            label = f"ask_{q.qtype}" + (f":{q.variable}" if q.variable else "")
            eus[label] = eu_ask(s.b, q, s.cat, self.ch, s.bucket, self.mode, self.costs, self.allow_abstain)
            if eus[label] > best + 1e-12:
                best, best_q = eus[label], q
        if best_q is not None:
            return Decision("ask", best_q, eus)
        return Decision("abstain" if (self.allow_abstain and eus["abstain"] >= eus["answer"]) else "answer", None, eus)
