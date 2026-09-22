"""Configuration loading.

One YAML file describes a whole run. Everything else in the codebase reads
from the object this module returns, so an experiment is reproducible from
`configs/*.yaml` plus the seed.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    """A thin typed wrapper around the YAML config."""

    raw: Dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    @property
    def seed(self) -> int:
        return int(self.raw.get("seed", 1234))

    def path(self, key: str) -> Path:
        """Resolve a key under `paths:` relative to the repository root."""
        value = self.raw["paths"][key]
        p = Path(value)
        return p if p.is_absolute() else REPO_ROOT / p


def load_config(path: str | Path = "configs/default.yaml") -> Config:
    """Load a YAML config and seed every random source we use."""
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    with open(p, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    cfg = Config(raw=raw)
    set_seed(cfg.seed)
    return cfg


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and (if present) torch."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        # torch is optional for the pure-analysis entry points.
        pass
