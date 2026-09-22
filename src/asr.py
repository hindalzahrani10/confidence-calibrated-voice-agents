"""Speech recognition with word-level confidence.

We use faster-whisper because it exposes a per-word probability, which is
the ASR half of our confidence signal bank. An end-to-end audio LLM can be
swapped in later; the cascade is used first precisely because it makes the
acoustic uncertainty visible as a number.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class ASRResult:
    """A transcript plus everything we need to score acoustic confidence."""

    text: str
    word_probs: List[float] = field(default_factory=list)
    avg_logprob: float = 0.0
    no_speech_prob: float = 0.0

    @property
    def mean_word_confidence(self) -> float:
        return float(np.mean(self.word_probs)) if self.word_probs else 0.0

    @property
    def min_word_confidence(self) -> float:
        return float(np.min(self.word_probs)) if self.word_probs else 0.0


class WhisperASR:
    """Thin wrapper over faster-whisper that always returns word probabilities."""

    def __init__(
        self,
        model_size: str = "small",
        device: str = "auto",
        compute_type: str = "default",
        language: str = "en",
        beam_size: int = 5,
    ) -> None:
        from faster_whisper import WhisperModel

        self.language = language
        self.beam_size = beam_size
        self.model = WhisperModel(
            model_size, device=device, compute_type=compute_type
        )

    def transcribe(self, audio_path: str) -> ASRResult:
        """Transcribe one file and collect per-word probabilities."""
        segments, _info = self.model.transcribe(
            audio_path,
            language=self.language,
            beam_size=self.beam_size,
            word_timestamps=True,
        )

        parts: List[str] = []
        word_probs: List[float] = []
        seg_logprobs: List[float] = []
        no_speech: List[float] = []

        # `segments` is a generator: consuming it is what runs the model.
        for seg in segments:
            parts.append(seg.text)
            seg_logprobs.append(float(seg.avg_logprob))
            no_speech.append(float(seg.no_speech_prob))
            for word in seg.words or []:
                word_probs.append(float(word.probability))

        return ASRResult(
            text="".join(parts).strip(),
            word_probs=word_probs,
            avg_logprob=float(np.mean(seg_logprobs)) if seg_logprobs else 0.0,
            no_speech_prob=float(np.mean(no_speech)) if no_speech else 0.0,
        )

    def nbest_disagreement(self, audio_path: str, n: int = 3) -> float:
        """A cheap word-error-rate proxy: how much do n decodes disagree?

        Returns a value in [0, 1], where 0 means every decode produced the
        same token sequence. Sampling is enabled via temperature fallback,
        so repeated calls on clean audio should agree closely.
        """
        decodes: List[str] = []
        for _ in range(n):
            segments, _ = self.model.transcribe(
                audio_path,
                language=self.language,
                beam_size=1,
                temperature=0.4,
                word_timestamps=False,
            )
            decodes.append("".join(s.text for s in segments).strip().lower())

        if not decodes:
            return 1.0

        token_sets = [set(d.split()) for d in decodes]
        union = set().union(*token_sets)
        if not union:
            return 0.0
        intersection = set(token_sets[0]).intersection(*token_sets[1:])
        return 1.0 - len(intersection) / len(union)
