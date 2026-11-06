"""EVALUATOR-SIDE lookup from a simulated user reply to its ASR transcript.

A reply (text, item, SNR) maps to a deterministic clip id with the same formula the Phase 2 generator
uses, so replies already in the Phase 2 inventory are reused and anything else becomes an explicit,
logged on-demand request that `scripts/generate_ondemand.py` fills (TTS + noise + ASR) before the run
completes. Transcripts are cached by clip id; a policy never triggers fresh randomness.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parent.parent


def reply_clip_id(master_seed: int, base_id: str, text: str, snr: Optional[int]) -> Tuple[str, str]:
    key = text.strip().lower()
    rkey = hashlib.sha256(f"{base_id}|{key}".encode()).hexdigest()[:8]
    tag = "clean" if snr is None else f"{snr}|0"
    return "r" + hashlib.sha256(f"{master_seed}|reply|{base_id}|{rkey}|{tag}".encode()).hexdigest()[:10], rkey


class MissingReply(Exception):
    pass


class ReplyService:
    def __init__(self, master_seed: int, asr_files: list, requests_path: Path) -> None:
        self.seed = master_seed
        self.asr: Dict[str, dict] = {}
        for f in asr_files:
            f = Path(f)
            if f.exists():
                for line in f.read_text().splitlines():
                    if line.strip():
                        r = json.loads(line)
                        self.asr[r["clip_id"]] = r
        self.requests_path = Path(requests_path)
        self.missing: Dict[str, dict] = {}

    def get(self, base_id: str, split: str, text: str, snr: Optional[int]) -> dict:
        cid, rkey = reply_clip_id(self.seed, base_id, text, snr)
        if cid in self.asr:
            return {**self.asr[cid], "clip_id": cid}
        self.missing[cid] = {"clip_id": cid, "base_id": base_id, "split": split, "text": text.strip(), "snr": snr,
                             "reply_key": rkey}
        raise MissingReply(cid)

    def flush_requests(self) -> int:
        self.requests_path.parent.mkdir(parents=True, exist_ok=True)
        have = set()
        if self.requests_path.exists():
            have = {json.loads(l)["clip_id"] for l in self.requests_path.read_text().splitlines() if l.strip()}
        new = [r for k, r in self.missing.items() if k not in have]
        with open(self.requests_path, "a") as fh:
            for r in new:
                fh.write(json.dumps(r, sort_keys=True) + "\n")
        return len(new)
