"""Controlled noise mixing with a measured, not nominal, SNR.

Why this replaces `data.add_noise` for Phase 2: that function takes signal
power over the whole clip (silence included), uses white noise only, and draws
from one shared random stream so results depend on processing order. Here the
SNR is defined on *active speech* power, noise is a real recording sliced by
a per-clip seed, and every number needed to rebuild a clip is logged.
"""

from __future__ import annotations

import hashlib
from typing import Tuple

import numpy as np

PCM_SCALE = 32767.0


def derive_seed(*parts) -> int:
    """Order-independent per-clip seed: no global RNG state is ever consumed."""
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(h[:8], "big")


def frame_energies(x: np.ndarray, sr: int, frame_ms: float, hop_ms: float) -> np.ndarray:
    frame, hop = int(sr * frame_ms / 1000), int(sr * hop_ms / 1000)
    if len(x) < frame:
        return np.array([float(np.mean(x.astype(np.float64) ** 2))])
    n = 1 + (len(x) - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    return np.mean(x.astype(np.float64)[idx] ** 2, axis=1)


def active_ms(x: np.ndarray, sr: int, frame_ms: float, hop_ms: float, dyn_db: float) -> float:
    """Mean square over frames within `dyn_db` of the loudest frame."""
    e = frame_energies(x, sr, frame_ms, hop_ms)
    emax = float(e.max())
    if emax <= 0.0:
        return 0.0
    return float(np.mean(e[e > emax * 10.0 ** (-dyn_db / 10.0)]))


def to_pcm16(x: np.ndarray) -> np.ndarray:
    return np.clip(np.round(x.astype(np.float64) * PCM_SCALE), -32768, 32767).astype(np.int16)


def from_pcm16(q: np.ndarray) -> np.ndarray:
    return (q.astype(np.float64) / PCM_SCALE).astype(np.float32)


def level_db(ms: float) -> float:
    return 10.0 * np.log10(max(ms, 1e-20))


def normalise_speech(x: np.ndarray, sr: int, target_db: float, al: dict) -> Tuple[np.ndarray, float]:
    """Scale so the active-speech level equals `target_db`; returns (scaled, gain)."""
    cur = level_db(active_ms(x, sr, al["frame_ms"], al["hop_ms"], al["dynamic_range_db"]))
    gain = 10.0 ** ((target_db - cur) / 20.0)
    return (x * gain).astype(np.float32), float(gain)


def draw_offset(n_samples: int, sr: int, part: Tuple[float, float], seed: int) -> int:
    """Start sample of a noise slice lying wholly inside the split's partition."""
    lo, hi = int(part[0] * sr), int(part[1] * sr) - n_samples
    if hi < lo:
        raise ValueError("noise partition shorter than the clip")
    return int(np.random.default_rng(seed).integers(lo, hi + 1))


def mix_at_snr(clean: np.ndarray, noise_seg: np.ndarray, snr_db: float, sr: int, al: dict):
    """Return (mixture_float, noise_gain, measured_snr_db).

    `clean` must already be the *quantised* clean waveform so the mixture can
    be rebuilt exactly from the stored files.
    """
    p_speech = active_ms(clean, sr, al["frame_ms"], al["hop_ms"], al["dynamic_range_db"])
    p_noise = float(np.mean(noise_seg.astype(np.float64) ** 2))
    if p_speech <= 0.0 or p_noise <= 0.0:
        raise ValueError("silent speech or noise segment")
    gain = float(np.sqrt(p_speech / (p_noise * 10.0 ** (snr_db / 10.0))))
    mix = clean.astype(np.float64) + gain * noise_seg.astype(np.float64)
    measured = 10.0 * np.log10(p_speech / (gain**2 * p_noise))
    return mix.astype(np.float32), gain, float(measured)
