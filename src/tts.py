"""Deterministic local text-to-speech (Piper) at the project sample rate.

The voice is synthetic. Sampling noise is switched off so identical text gives
identical samples within one software environment; across machines only a
waveform-correlation tolerance is a realistic expectation (ONNX runtimes may
differ in the last bits), which is why every output's hash is recorded.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly


class PiperTTS:
    def __init__(self, model_path: str | Path, cfg: dict, target_sr: int = 16000) -> None:
        from piper import PiperVoice, SynthesisConfig

        self.voice = PiperVoice.load(str(model_path))
        self.native_sr = int(self.voice.config.sample_rate)
        self.target_sr = target_sr
        self.syn_cfg = SynthesisConfig(
            length_scale=cfg["length_scale"],
            noise_scale=cfg["noise_scale"],
            noise_w_scale=cfg["noise_w_scale"],
            normalize_audio=cfg["normalize_audio"],
        )
        ratio = Fraction(target_sr, self.native_sr)
        self.up, self.down = ratio.numerator, ratio.denominator

    def synthesize(self, text: str) -> np.ndarray:
        """Float32 mono audio at `target_sr`."""
        chunks = [c.audio_float_array for c in self.voice.synthesize(text, syn_config=self.syn_cfg)]
        if not chunks:
            raise ValueError(f"TTS produced no audio for {text!r}")
        x = np.concatenate(chunks)
        return resample_poly(x, self.up, self.down).astype(np.float32)
