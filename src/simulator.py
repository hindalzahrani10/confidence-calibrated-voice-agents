"""EVALUATOR-SIDE question-conditioned user simulator (Phase 1 contract).

This is the only component that holds the hidden task. The agent submits a structured question;
the simulator returns what the user would SAY, given that question and the hidden task:

  open     -> the user restates the information that was still unresolved (Phase 1 full-coverage
              reply), or says it does not understand when nothing was unresolved.
  closed   -> the offered option equal to the hidden value (echoing the agent's own wording), else
              "None of those." -- never the nearest wrong option, and no value is revealed.
  confirm  -> "Yes." only for a proposal equal to an acceptable task; otherwise "No." or, in the
              corrective style, "No, <hidden value(s)>". A bare no never reveals the task.

Whether a wrong proposal gets a bare or corrective no is a recorded setting
(`style`: 'mixed' = seeded 50/50 per (item, proposal), 'bare', 'corrective').
"""

from __future__ import annotations

import hashlib
from typing import Dict, Optional, Tuple

from .nmvoi import Question, value_of
from .ontology import Task, display, intent_phrase

OFF_TARGET = "I'm not sure what you mean."


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:]


class UserSimulator:
    def __init__(self, record: dict, style: str = "mixed", seed: int = 1234) -> None:
        P = record["private"]
        self.base_id = record["base_id"]
        self.gold = Task.make(P["gold_task"]["intent"], P["gold_task"]["slots"])
        self.acceptable = P["acceptable_tasks"]
        self.say: Dict[str, str] = P["say"]
        self.required = P["required_slots"]
        self.known = {k for k in P["utterance_slots"] if k not in P["at_risk_slots"]}
        opens = [p for p in P["probes"] if p["question_type"] == "open"]
        self.open_text: Optional[str] = max(opens, key=lambda p: len(p["covers"]))["reply"]["text"] if opens else None
        self.style, self.seed = style, seed

    def accepts(self, task: Task) -> bool:
        for pat in self.acceptable:
            if task.intent == pat["intent"] and set(task.sdict) == set(pat["slots"]) and all(
                    v == "*" or task.sdict[k] == v for k, v in pat["slots"].items()):
                return True
        return False

    def _phrase(self, key: str) -> str:
        if key in self.say:
            return self.say[key]
        return intent_phrase(self.gold.intent) if key == "intent" else display(key, self.gold.sdict.get(key, ""))

    def reply(self, q: Question, cat) -> Tuple[str, str]:
        """(text the user says, reply type)."""
        if q.qtype == "open":
            return (self.open_text, "answer") if self.open_text else (OFF_TARGET, "off_target")
        if q.qtype == "closed":
            gv = value_of(self.gold, q.variable)
            if gv is None:
                return OFF_TARGET, "off_target"
            if gv in q.option_values:
                return _cap(q.option_texts[q.option_values.index(gv)]), "select"
            return "None of those.", "none_of_these"
        proposal: Task = cat[q.proposal]
        if self.accepts(proposal):
            return "Yes.", "yes"
        corrective = self.style == "corrective"
        if self.style == "mixed":
            h = hashlib.sha256(f"{self.seed}|{self.base_id}|{proposal.key()}".encode()).digest()[0]
            corrective = h % 2 == 1
        if not corrective:
            return "No.", "no_bare"
        diff = [k for k, v in proposal.sdict.items() if k in self.gold.sdict and self.gold.sdict[k] != v]
        if proposal.intent != self.gold.intent:
            diff.append("intent")
        if not diff:
            return "No.", "no_bare"
        return "No, " + " and ".join(self._phrase(k) for k in diff) + ".", "no_correction"
