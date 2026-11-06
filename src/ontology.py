"""Agent-side task ontology and lexical belief model (NO gold labels, NO dataset-private fields).

What the agent knows (public, shared by every policy):
  * PATTERNS: for each intent, the words that express it and where its slots go.
  * A catalogue of candidate tasks: per (intent, slot-name set), the cross-product
    of the slot values seen anywhere in the Phase 1 value inventory. The inventory is
    pooled over all items WITHOUT item association, so the catalogue always contains
    the gold task of every item (label-space leakage, identical for all policies, and
    recorded as a threat to validity). Gold/alternative roles are not available here.

Belief: score each candidate task by how well it explains the ASR transcript
(weighted by ASR word confidence), then softmax with a temperature fitted on the
TRAIN split only. Unspoken slots earn no credit and no penalty, so underspecified
requests leave several completions tied, which is exactly the uncertainty a
clarification should resolve.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------- text ----------
ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
STOP = set("a an the to for of in on at that it is what whats what's my your me you i i'm this and by with from".split())
ALIAS = {"tonight": "today", "p.m.": "pm", "a.m.": "am", "mum": "mom", "mama": "mom", "okay": "ok",
         "track": "song", "tracks": "song", "songs": "song"}   # generic synonyms only


def num_words(n: int) -> List[str]:
    if n < 20:
        return [ONES[n]]
    if n < 100:
        return [TENS[n // 10]] + ([ONES[n % 10]] if n % 10 else [])
    if n < 1000:
        return [ONES[n // 100], "hundred"] + (num_words(n % 100) if n % 100 else [])
    return [str(n)]


def word_tokens(w: str) -> List[str]:
    """Normalise ONE ASR/text word into 0+ tokens (digits and times become number words)."""
    w = w.lower().strip()
    w = ALIAS.get(w, w)
    m = re.fullmatch(r"(\d{1,2}):(\d{2})([ap]m)?", w)
    if m:
        out = num_words(int(m.group(1)))
        if m.group(2) != "00":
            out += (["oh"] if m.group(2)[0] == "0" else []) + num_words(int(m.group(2)))
        return out + ([m.group(3)] if m.group(3) else [])
    w = re.sub(r"^[^\w']+|[^\w']+$", "", w)          # strip edge punctuation
    w = re.sub(r"'s$", "", w)                         # possessive
    if "-" in w:
        return [t for p in w.split("-") for t in word_tokens(p)]
    if re.fullmatch(r"\.?\d+", w) or re.fullmatch(r"\d+", w):
        return num_words(int(w.lstrip(".")))
    w = ALIAS.get(w, w)
    return [w] if w else []


def tokens(text: str) -> List[str]:
    text = text.lower().replace("in the morning", "am").replace("in the evening", "pm").replace("at night", "pm")
    text = re.sub(r"(\d)\s*[:.]\s*(\d{2})\b", r"\1:\2", text)
    return [t for w in text.split() for t in word_tokens(w)]


def sim(a: str, b: str) -> float:
    """1 - normalised edit distance, a crude stand-in for acoustic confusability."""
    if a == b:
        return 1.0
    n, m = len(a), len(b)
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1]))
        prev = cur
    return 1.0 - prev[m] / max(n, m)


def match(a: str, b: str, theta: float = 0.6) -> float:
    s = sim(a, b)
    return s if s >= theta else 0.0


# ------------------------------------------------------------------------- patterns --------
# "[...]" groups are dropped when their slot is not part of the task's slot set.
PATTERNS: Dict[str, str] = {
    "set_alarm": "set an alarm for {time}", "light_off": "turn off the {room} lights",
    "get_forecast": "get the weather [in {city}] [for {date}]", "next_track": "skip to the next track",
    "previous_track": "go back to the previous track", "pause": "pause the music", "stop_music": "stop the music",
    "play_music": "start the music", "set_timer": "set a timer for {duration_min} minutes",
    "add_item": "add {item} to your {list} list", "call_contact": "call {contact}",
    "set_volume": "set the volume to {level} percent", "get_time": "tell you the time in {city}",
    "create_reminder": "set a reminder for {task} [at {time}] [{date}] [{lead_time_min} minutes before]",
    "send_text": "text {recipient} [saying {body}]", "navigate": "navigate to [{destination}] [the {place}]",
    "set_brightness": "set the brightness to {level} percent", "play_artist": "play music by {artist}",
    "dim_lights": "dim the {room} lights", "unlock_door": "unlock the {door} door", "lock_door": "lock the {door} door",
    "open_door": "open the {door} door", "play_song": "play the song {title}",
    "play_music_video": "play the music video for {title}", "play_album": "play the album {title}",
    "create_calendar_event": "add a calendar event for {task} {date} at {time}",
    "create_note": "save a note about {task} {date} at {time}", "record_audio": "record audio only",
    "record_video": "record video", "record_screen": "record the screen",
    "cancel_food_delivery": "cancel your {date} food delivery", "cancel_shopping_order": "cancel your {date} shopping order",
    "cancel_restaurant_reservation": "cancel your {date} restaurant reservation", "call_business": "call the {place}",
    "show_business_hours": "show opening hours for the {place}",
    "set_thermostat": "set the [{room}] temperature to [{value}] [{temperature_c} degrees]",
    "set_light_brightness": "set the {room} lights to {value} percent",
    "set_speaker_volume": "set the {room} speaker volume to {value} percent",
    "book_hotel": "book a hotel in {destination} for {month}", "book_flight": "book a flight to {destination} for {month}",
    "book_rental_car": "book a rental car in {destination} for {month}", "show_calendar": "show your calendar for {date}",
    "show_reminders": "show your reminders for {date}", "show_contact_details": "show {contact} contact details",
    "open_chat": "open a chat with {contact}", "navigate_to_event": "navigate to the {event} {date}",
    "play_event_broadcast": "put on the {event} {date}", "buy_event_tickets": "buy tickets to the {event} {date}",
    "dismiss_alarm": "dismiss the ringing alarm", "disable_alarm": "disable the alarm",
    "disarm_security_system": "disarm the security system", "mute_chat_notifications": "mute {person} chat notifications",
    "mute_call_participant": "mute {person} on the call", "mute_social_feed": "mute {person} posts",
    "play_genre": "play some {genre}", "book_table": "book a table for {party_size} at {time} {date}",
    "adjust_volume": "turn the volume {direction} on the {device}", "cancel_appointment": "cancel the {appointment}",
    "book_ticket": "book {quantity} ticket to the {event}", "send_email": "email {attachment} to {recipient}",
}


def display(slot: str, value) -> str:
    if slot == "time" and re.fullmatch(r"\d\d:\d\d", str(value)):
        hh, mm = (int(x) for x in str(value).split(":"))
        h = hh % 12 or 12
        return f"{h}:{mm:02d} {'am' if hh < 12 else 'pm'}" if mm else f"{h} {'am' if hh < 12 else 'pm'}"
    return str(value)


def intent_phrase(intent: str) -> str:
    """Slot-free wording of an intent, e.g. 'play the music video' (agent-side option text)."""
    p = re.sub(r"\[.*?\]|\{\w+\}", " ", PATTERNS[intent])
    words = p.split()
    while words and words[-1] in STOP:
        words.pop()
    return " ".join(words)


@dataclass(frozen=True)
class Task:
    intent: str
    slots: Tuple[Tuple[str, object], ...]

    @classmethod
    def make(cls, intent: str, slots: dict) -> "Task":
        return cls(intent, tuple(sorted(slots.items())))

    @property
    def sdict(self) -> dict:
        return dict(self.slots)

    def key(self) -> str:
        return json.dumps([self.intent, self.slots], sort_keys=True, default=str)

    def render(self) -> str:
        """Imperative surface form, used both as scoring target and as the agent's proposal wording."""
        s = self.sdict

        def fill(m):
            inner = m.group(1)
            names = re.findall(r"\{(\w+)\}", inner)
            return inner if all(n in s for n in names) else ""
        p = re.sub(r"\[(.*?)\]", fill, PATTERNS[self.intent])
        p = re.sub(r"\{(\w+)\}", lambda m: display(m.group(1), s.get(m.group(1), "")), p)
        return re.sub(r"\s+", " ", p).strip()

    def parts(self) -> Tuple[List[str], List[str]]:
        """(template content tokens, slot content tokens)."""
        s = self.sdict
        slot_toks: List[str] = []
        tmpl_toks: List[str] = []
        p = re.sub(r"\[(.*?)\]", lambda m: m.group(1) if all(n in s for n in re.findall(r"\{(\w+)\}", m.group(1))) else "",
                   PATTERNS[self.intent])
        for piece in re.split(r"(\{\w+\})", p):
            mm = re.fullmatch(r"\{(\w+)\}", piece)
            if mm:
                slot_toks += [t for t in tokens(display(mm.group(1), s[mm.group(1)])) if t not in STOP]
            else:
                tmpl_toks += [t for t in tokens(piece) if t not in STOP]
        return tmpl_toks, slot_toks


def load_catalogue(path) -> List[Task]:
    """Load the public candidate-task list (written by scripts/build_catalogue.py; no roles, no item ids)."""
    return [Task.make(t["intent"], t["slots"]) for t in json.loads(Path(path).read_text())["tasks"]]


# --------------------------------------------------------------------------- belief --------
LAM = {"explain": 1.0, "unexplained": 1.5, "slot": 1.0, "template": 0.5}


def observe(words: Sequence[dict]) -> List[Tuple[str, float]]:
    """ASR words -> content tokens with confidence weights (digits/times normalised)."""
    out = []
    for w in words:
        for t in word_tokens(w["w"]):
            if t not in STOP:
                out.append((t, max(float(w["p"]), 0.05)))
    return out


class BeliefModel:
    """P(y | transcript) over a fixed catalogue; `beta` is the only fitted parameter."""

    def __init__(self, catalogue: List[Task], beta: float = 1.0) -> None:
        self.cat = catalogue
        self.beta = beta
        self._parts = [t.parts() for t in catalogue]

    def jitter(self, key: str) -> np.ndarray:
        """Tiny hash-derived offsets (<=1e-6) that break EXACT ties without reference to any label.

        Without this, ties fall back on catalogue order (alphabetical), which can coincide with the true
        task and reward answering a genuinely ambiguous request by luck.
        """
        return 1e-6 * np.array([int.from_bytes(hashlib.sha256(f"{key}|{t.key()}".encode()).digest()[:4], "big") / 2**32
                                for t in self.cat])

    def scores(self, obs: List[Tuple[str, float]], tie_key: str | None = None) -> np.ndarray:
        out = np.zeros(len(self.cat))
        for k, (tm, st) in enumerate(self._parts):
            ybag = tm + st
            s = 0.0
            for tok, w in obs:
                ex = max((match(tok, c) for c in ybag), default=0.0)
                s += LAM["explain"] * w * ex - LAM["unexplained"] * w * (1.0 - ex)
            for c in st:
                s += LAM["slot"] * max((w * match(c, tok) for tok, w in obs), default=0.0)
            for c in tm:
                s += LAM["template"] * max((w * match(c, tok) for tok, w in obs), default=0.0)
            out[k] = s
        return out + self.jitter(tie_key) if tie_key else out

    def posterior(self, obs: List[Tuple[str, float]], beta: float | None = None, tie_key: str | None = None) -> np.ndarray:
        z = (self.beta if beta is None else beta) * self.scores(obs, tie_key)
        z -= z.max()
        p = np.exp(z)
        return p / p.sum()
