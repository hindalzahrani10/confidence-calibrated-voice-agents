"""Build the spoken evaluation set.

Three sources, combined into one labelled JSONL:

  1. real spoken commands (downloaded, not committed)
  2. text ambiguity items voiced with TTS
  3. a hand-written underspecification set

Then every item is expanded into noise variants at the SNR levels in the
config, and split into validation and test. The test split is frozen here
and must never be used for threshold tuning.

    python scripts/build_data.py --config configs/default.yaml

The TTS step is the one piece that needs a voice model. If you have not
installed one yet, run with --skip-tts to build the real-audio half only
and get the pipeline moving; add the synthesized half afterwards.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config  # noqa: E402
from src.data import EvalItem, make_noise_variants, write_labels  # noqa: E402

# A small hand-written underspecification set: each command is clean,
# unambiguous English, but missing exactly one argument the agent needs.
# Grow this to 50-100 items; it is the cheapest data you will ever collect.
UNDERSPECIFIED = [
    ("set an alarm", "What time should I set it for?", "seven in the morning"),
    ("remind me about the meeting", "When should I remind you?", "an hour before"),
    ("play some music", "What would you like to hear?", "jazz"),
    ("book a table", "For how many people and when?", "two people at eight"),
    ("send a message to Sam", "What should the message say?", "running late"),
    ("turn up the volume", "On which device?", "the living room speaker"),
    ("add milk to the list", "Which list?", "the grocery list"),
    ("cancel it", "Which item should I cancel?", "the dentist appointment"),
]


def build_underspecified_items(audio_dir: Path, skip_tts: bool) -> list[EvalItem]:
    """Voice the underspecification set and wrap it as EvalItems."""
    items: list[EvalItem] = []
    audio_dir.mkdir(parents=True, exist_ok=True)

    for i, (text, question, reply) in enumerate(UNDERSPECIFIED):
        path = audio_dir / f"underspec_{i:03d}.wav"
        if not skip_tts and not path.exists():
            synthesize(text, path)
        items.append(
            EvalItem(
                item_id=f"underspec_{i:03d}",
                audio_path=str(path),
                reference_text=text,
                ambiguity_type="underspecified",
                gold_answers=[reply],
                clarification_reply=reply,
                snr_db=None,
                source="self_built",
                meta={"expected_question": question},
            )
        )
    return items


def synthesize(text: str, out_path: Path) -> None:
    """Render `text` to a 16 kHz wav.

    Uses Piper if it is installed. Any TTS works here; the only requirement
    is that the same voice is used for the whole controlled set, so that
    voice quality is not confounded with ambiguity type.
    """
    try:
        import subprocess

        subprocess.run(
            ["piper", "--model", "en_US-lessac-medium", "--output_file", str(out_path)],
            input=text.encode("utf-8"),
            check=True,
            capture_output=True,
        )
    except (FileNotFoundError, ImportError) as exc:
        raise SystemExit(
            f"No TTS available ({exc}). Install piper-tts, or re-run with "
            "--skip-tts to build the real-audio half first."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--skip-tts", action="store_true",
                        help="Skip synthesis; useful for a first smoke test.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    rng = np.random.default_rng(cfg.seed)
    audio_dir = cfg.path("audio_dir")

    items = build_underspecified_items(audio_dir, args.skip_tts)
    print(f"Base items: {len(items)}")
    print(
        "TODO for you: add real audio (SLURP/VoiceBench) and voiced AmbigQA /\n"
        "ClarQ-LLM items here. src.data.load_slurp_subset is the starting point."
    )

    # Expand into noise variants, unless we skipped synthesis (no audio yet).
    if not args.skip_tts:
        expanded: list[EvalItem] = []
        for item in items:
            expanded.extend(
                make_noise_variants(
                    item, cfg["data"]["snr_levels"], audio_dir / "noisy", rng
                )
            )
        items = expanded
        print(f"After noise variants: {len(items)}")

    # Split by BASE item, so that noise variants of the same utterance never
    # straddle the val/test boundary and leak information.
    base_ids = sorted({item.meta.get("derived_from", item.item_id) for item in items})
    rng.shuffle(base_ids)
    n_val = max(1, int(len(base_ids) * cfg["data"]["val_fraction"]))
    val_ids = set(base_ids[:n_val])

    for item in items:
        base = item.meta.get("derived_from", item.item_id)
        item.split = "val" if base in val_ids else "test"

    labels_path = cfg.path("labels")
    write_labels(items, labels_path)

    n_val_items = sum(1 for i in items if i.split == "val")
    print(f"\nWrote {labels_path}")
    print(f"  val:  {n_val_items} items")
    print(f"  test: {len(items) - n_val_items} items (frozen; never tune on these)")


if __name__ == "__main__":
    main()
