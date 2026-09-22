"""Evaluation-set construction.

The central experimental control of this project is being able to dial
acoustic uncertainty up and down *independently* of intent ambiguity. That
is why the evaluation set has two halves:

  * real audio   : SLURP slices, genuine spoken assistant commands
  * controlled   : text ambiguity benchmarks voiced with TTS, then
                   corrupted with additive noise at known SNR levels

Every item carries an `ambiguity_type` label, a gold answer, and a scripted
reply that the simulated user gives if the agent asks a clarifying question.

No large files are committed: loaders download public datasets on demand.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000

# The four uncertainty classes the project is built around.
AMBIGUITY_TYPES = ("none", "acoustic", "intent", "underspecified")


@dataclass
class EvalItem:
    """One utterance in the evaluation set."""

    item_id: str
    audio_path: str
    reference_text: str
    ambiguity_type: str
    gold_answers: List[str]
    # What the simulated user says back if the agent asks for clarification.
    clarification_reply: str = ""
    # None means clean audio; otherwise the SNR in dB that was applied.
    snr_db: Optional[float] = None
    source: str = "unknown"
    split: str = "val"
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.ambiguity_type not in AMBIGUITY_TYPES:
            raise ValueError(
                f"unknown ambiguity_type {self.ambiguity_type!r}; "
                f"expected one of {AMBIGUITY_TYPES}"
            )


def add_noise(audio: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Mix white noise into `audio` at a target signal-to-noise ratio.

    Returns a copy scaled to stay inside [-1, 1]. This is deliberately the
    simplest possible corruption: swap in recorded babble or MUSAN noise
    once the pipeline works end to end.
    """
    audio = audio.astype(np.float32)
    signal_power = float(np.mean(audio**2))
    if signal_power <= 0.0:
        return audio

    noise = rng.normal(0.0, 1.0, size=audio.shape).astype(np.float32)
    noise_power = float(np.mean(noise**2))
    target_noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    noise *= np.sqrt(target_noise_power / noise_power)

    noisy = audio + noise
    peak = float(np.max(np.abs(noisy)))
    if peak > 1.0:
        noisy = noisy / peak
    return noisy


def make_noise_variants(
    item: EvalItem,
    snr_levels: List[Optional[float]],
    out_dir: Path,
    rng: np.random.Generator,
) -> List[EvalItem]:
    """Write one noise-corrupted copy of `item` per SNR level.

    A `None` entry in `snr_levels` yields the clean item unchanged. Items
    that gain noise are relabelled `acoustic` only if they were previously
    `none`: a question that was already intent-ambiguous stays that way.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    audio, sr = sf.read(item.audio_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    variants: List[EvalItem] = []
    for snr in snr_levels:
        if snr is None:
            variants.append(item)
            continue

        noisy = add_noise(audio, float(snr), rng)
        variant_id = f"{item.item_id}__snr{int(snr)}"
        path = out_dir / f"{variant_id}.wav"
        sf.write(path, noisy, sr)

        variants.append(
            EvalItem(
                item_id=variant_id,
                audio_path=str(path),
                reference_text=item.reference_text,
                ambiguity_type=(
                    "acoustic" if item.ambiguity_type == "none" else item.ambiguity_type
                ),
                gold_answers=list(item.gold_answers),
                clarification_reply=item.clarification_reply,
                snr_db=float(snr),
                source=item.source,
                split=item.split,
                meta=dict(item.meta, derived_from=item.item_id),
            )
        )
    return variants


def load_slurp_subset(n: int = 150, split: str = "test") -> List[dict]:
    """Fetch a small SLURP slice (real spoken assistant commands).

    Returns raw records; converting them to `EvalItem`s needs the audio
    written to disk, which `scripts/build_data.py` does. Kept separate so
    this module imports cleanly without network access.
    """
    from datasets import load_dataset

    ds = load_dataset("qanastek/MASSIVE", "en-US", split=split, streaming=True)
    records = []
    for i, row in enumerate(ds):
        if i >= n:
            break
        records.append(row)
    return records


def write_labels(items: List[EvalItem], path: Path) -> None:
    """Persist the evaluation set as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for item in items:
            fh.write(json.dumps(asdict(item)) + "\n")


def read_labels(path: Path) -> List[EvalItem]:
    """Load the evaluation set back from JSONL."""
    items: List[EvalItem] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                items.append(EvalItem(**json.loads(line)))
    return items


def iter_split(items: List[EvalItem], split: str) -> Iterator[EvalItem]:
    """Yield only the items belonging to `split` ("val" or "test").

    Tuning anything on "test" invalidates every headline number in the
    paper, so the split lives on the item itself rather than in a notebook.
    """
    for item in items:
        if item.split == split:
            yield item
