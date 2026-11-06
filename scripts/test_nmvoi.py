"""Unit tests for the NM-VoI machinery: analytic checks, monotonicity in reliability and cost,
SNR dependence that the ablations must NOT have, and gold-label leakage guards.

    python scripts/test_nmvoi.py
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import nmvoi as V  # noqa: E402
from src.channel import ChannelTable  # noqa: E402
from src.ontology import Task  # noqa: E402

FAILS = []


def check(ok, msg):
    print(("ok   " if ok else "FAIL ") + msg)
    if not ok:
        FAILS.append(msg)


def table(r_by_bucket: dict, q="closed", other=0.9):
    counts = {k: {b: [int(round(other * 1000)), 1000] for b in ("clean", "20", "10", "5")}
              for k in ("confirm_yes", "confirm_no", "closed", "open")}
    counts[q] = {b: [int(round(r * 1000)), 1000] for b, r in r_by_bucket.items()}
    return ChannelTable(counts, prior_strength=0.0001)


names = ["Mom", "Tom", "Tim", "Don"]
cat = [Task.make("call_contact", {"contact": n}) for n in names]
b_unif = np.full(4, 0.25)
COSTS = V.Costs()

# 1. analytic values -----------------------------------------------------------------------
check(abs(V.eu_answer(b_unif) - (-0.5)) < 1e-12, "EU(answer) = 2*max(b)-1 = -0.5 for a uniform 4-way belief")
perfect = table({b: 0.999999 for b in ("clean", "20", "10", "5")}, "closed")
q = V.closed_question(cat, b_unif, "contact", 3)
eu = V.eu_ask(b_unif, q, cat, perfect, "clean", "snr", COSTS, True)
check(abs(eu - (1.0 - COSTS.closed)) < 1e-3, f"perfect channel closed question: EU = 1 - C_closed ({eu:.4f})")
P, _ = V.outcome_model(cat, b_unif, q, perfect, "clean", "snr")
check(np.allclose(P.sum(axis=1), 1.0), "channel rows are proper probability distributions")
useless = table({b: 1.0 / 4 for b in ("clean", "20", "10", "5")}, "closed")  # r = 1/|O|: reply carries no information
eu0 = V.eu_ask(b_unif, q, cat, useless, "clean", "snr", COSTS, True)
check(eu0 < -COSTS.closed + 1e-6 and eu0 <= 0.0 + 1e-9, f"uninformative channel gives no value, only cost ({eu0:.3f})")

# 2. monotone in reliability, flips from ask to not-ask ---------------------------------------
vals = []
for r in (0.99, 0.8, 0.6, 0.4, 0.3):
    t = table({b: r for b in ("clean", "20", "10", "5")}, "closed")
    vals.append(V.eu_ask(b_unif, q, cat, t, "clean", "snr", COSTS, True))
check(all(a >= b - 1e-9 for a, b in zip(vals, vals[1:])), f"EU(ask) is non-increasing as reliability drops {np.round(vals, 3).tolist()}")
pol = lambda t, mode="snr": V.VoI("x", t, mode, COSTS, allowed=("closed",))  # noqa: E731
b_mod = np.array([0.55, 0.25, 0.1, 0.1])
kinds = []
for r in (0.99, 0.9, 0.7, 0.5, 0.35):
    t = table({b: r for b in ("clean", "20", "10", "5")}, "closed")
    kinds.append(pol(t).decide(V.AgentState(cat, b_mod, "clean")).kind)
check(kinds[0] == "ask" and kinds[-1] != "ask", f"decision flips from ask to not-ask as the channel degrades {kinds}")

# 3. monotone in cost --------------------------------------------------------------------------
t_good = table({b: 0.95 for b in ("clean", "20", "10", "5")}, "closed")
ks = [V.VoI("x", t_good, "snr", V.Costs(open=c, closed=c, confirm=c), allowed=("closed",)).decide(
    V.AgentState(cat, b_mod, "clean")).kind for c in (0.05, 0.2, 0.5, 1.5)]
check(ks[0] == "ask" and ks[-1] != "ask", f"decision flips from ask to not-ask as clarification cost rises {ks}")

# 4. SNR dependence: only NM-VoI may change with the bucket --------------------------------------
t_snr = table({"clean": 0.97, "20": 0.95, "10": 0.6, "5": 0.3}, "closed")
st = lambda bucket: V.AgentState(cat, b_mod, bucket)  # noqa: E731
nm = V.VoI("nm", t_snr, "snr", COSTS, allowed=("closed",))
ni = V.VoI("ni", t_snr, "pooled", COSTS, allowed=("closed",))
ev = V.VoI("ev", t_snr, "perfect", COSTS, allowed=("closed",))
check(nm.decide(st("clean")).kind == "ask" and nm.decide(st("5")).kind != "ask",
      "NM-VoI asks in clean audio but not at 5 dB (reliability collapses)")
check(len({ni.decide(st(b)).kind for b in ("clean", "20", "10", "5")}) == 1, "noise-independent VoI cannot change with SNR")
check(len({ev.decide(st(b)).kind for b in ("clean", "20", "10", "5")}) == 1, "standard EVPI cannot change with SNR")
check(abs(nm.decide(st("5")).eus["ask_closed:contact"] - ev.decide(st("5")).eus["ask_closed:contact"]) > 0.05,
      "NM-VoI and EVPI value the same question differently under noise")

# 5. posterior update semantics -------------------------------------------------------------------
tq = V.Question("confirm", proposal=0)
Pc, _ = V.outcome_model(cat, b_unif, tq, table({b: 0.9 for b in ("clean", "20", "10", "5")}, "confirm_yes", 0.9), "clean", "snr")
b_no = V.update(b_unif, Pc, 1)   # reply recognised as "no"
check(b_no[0] < 0.1 and abs(b_no.sum() - 1) < 1e-9, f"a 'no' lowers the proposal but does not identify another task (P(Mom)={b_no[0]:.3f}, others={np.round(b_no[1:], 3).tolist()})")
check(float(b_no.max()) < 0.5, "after a bare 'no' no single remaining task is certain (open-world exclusion, not recovery)")

# 6. leakage guards ----------------------------------------------------------------------------------
fields = set(V.AgentState.__dataclass_fields__)
check(fields == {"cat", "b", "bucket"}, f"AgentState carries only the catalogue, belief and SNR bucket {sorted(fields)}")
import ast


def code_names(module) -> set:
    """Identifiers and non-docstring string constants in a module's code (docstrings/comments excluded)."""
    tree = ast.parse(inspect.getsource(module))
    doc_ids = {id(n.body[0].value) for n in ast.walk(tree)
               if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef)) and n.body
               and isinstance(n.body[0], ast.Expr) and isinstance(getattr(n.body[0], "value", None), ast.Constant)}
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Name):
            out.add(n.id)
        elif isinstance(n, ast.Attribute):
            out.add(n.attr)
        elif isinstance(n, ast.arg):
            out.add(n.arg)
        elif isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            out.add(n.name)
        elif isinstance(n, ast.keyword) and n.arg:
            out.add(n.arg)
        elif isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in doc_ids:
            out.add(n.value)
    return {x.lower() for x in out}


names = set()
for m in (V, sys.modules["src.channel"], sys.modules["src.ontology"]):
    names |= code_names(m)
for tok in ("gold", "acceptable", "alternatives", "reference_text", "snr_db", "ambiguity_type", "hidden", "private"):
    hit = [n for n in names if tok in n]
    check(not hit, f"agent-side code has no identifier/string containing '{tok}' {hit[:3]}")

print(f"\n{'ALL PASSED' if not FAILS else str(len(FAILS)) + ' FAILED'}")
sys.exit(1 if FAILS else 0)
