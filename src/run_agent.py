"""Run the voice agent over the evaluation set and log everything.

This script does NOT decide anything final. It runs both branches for every
item (answer directly, and also answer after clarifying) and writes all five
confidence signals plus both outcomes to JSONL. Policy comparison then
becomes pure post-processing in `evaluate.py`, which means you can add a
sixth policy later without re-running the models.

Usage:
    python -m src.run_agent --config configs/default.yaml --limit 20
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm

from .asr import WhisperASR
from .config import load_config
from .data import read_labels
from .llm import LLMAgent
from .signals import attribute_uncertainty_source, build_signals


def is_correct(prediction: str, gold_answers: List[str]) -> bool:
    """Lenient containment match against any gold answer.

    Good enough to get the pipeline running and honest about its own
    crudeness. Upgrade to an LLM judge once the plumbing works, and report
    which scorer produced the headline numbers.
    """
    pred = prediction.strip().lower()
    if not pred:
        return False
    return any(g.strip().lower() in pred for g in gold_answers if g.strip())


def run_item(item, asr: WhisperASR, agent: LLMAgent, cfg) -> Dict:
    """Run one utterance through both branches and collect every signal."""
    started = time.time()

    asr_result = asr.transcribe(item.audio_path)
    transcript = asr_result.text

    # Branch 1: answer directly.
    answer_gen = agent.answer(transcript)

    # Confidence signals, all computed from the direct-answer branch so that
    # policies are compared on identical information.
    samples = agent.sample_answers(
        transcript,
        k=cfg["llm"]["n_samples"],
        temperature=cfg["llm"]["sample_temperature"],
    )
    verbalized = agent.verbalized_confidence(transcript, answer_gen.text)
    self_flag, self_reason = agent.self_interrogate(transcript)
    disagreement = asr.nbest_disagreement(item.audio_path)

    signals = build_signals(
        asr_result=asr_result,
        answer_gen=answer_gen,
        sampled_answers=[g.text for g in samples],
        verbalized=verbalized,
        self_clarify_flag=self_flag,
        nbest_disagreement=disagreement,
    )

    # Branch 2: ask one clarifying question, then answer. Running this for
    # every item (not only the ones a policy would ask about) is what lets
    # us label "clarification would have helped" for the fusion policy.
    question = agent.clarifying_question(transcript)
    clarified_gen = agent.answer_with_clarification(
        transcript, question, item.clarification_reply
    )

    direct_correct = is_correct(answer_gen.text, item.gold_answers)
    clarified_correct = is_correct(clarified_gen.text, item.gold_answers)

    return {
        "item_id": item.item_id,
        "split": item.split,
        "ambiguity_type": item.ambiguity_type,
        "snr_db": item.snr_db,
        "source": item.source,
        "reference_text": item.reference_text,
        "transcript": transcript,
        "direct_answer": answer_gen.text,
        "direct_correct": direct_correct,
        "clarifying_question": question,
        "clarification_reply": item.clarification_reply,
        "clarified_answer": clarified_gen.text,
        "clarified_correct": clarified_correct,
        # The fusion target: did asking actually buy anything?
        "clarification_helps": (not direct_correct) and clarified_correct,
        "signals": signals.to_dict(),
        "predicted_source": attribute_uncertainty_source(signals),
        "self_reason": self_reason,
        "latency_s": round(time.time() - started, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N items. Use --limit 20 for the pilot.",
    )
    parser.add_argument("--out", default=None, help="Output JSONL path.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    items = read_labels(cfg.path("labels"))
    if args.limit:
        items = items[: args.limit]

    print(f"Loaded {len(items)} items. Loading models (this takes a minute)...")
    asr = WhisperASR(
        model_size=cfg["asr"]["model"],
        device=cfg["asr"]["device"],
        compute_type=cfg["asr"]["compute_type"],
        language=cfg["asr"]["language"],
        beam_size=cfg["asr"]["beam_size"],
    )
    agent = LLMAgent(
        model_name=cfg["llm"]["model"],
        device_map=cfg["llm"]["device_map"],
        dtype=cfg["llm"]["dtype"],
        max_new_tokens=cfg["llm"]["max_new_tokens"],
        prompts_dir=cfg.path("prompts_dir"),
    )

    results_dir = cfg.path("results_dir")
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else results_dir / "runs.jsonl"

    n_failed = 0
    with open(out_path, "w", encoding="utf-8") as fh:
        for item in tqdm(items, desc="running agent"):
            try:
                record = run_item(item, asr, agent, cfg)
            except Exception as exc:  # keep going; a dead item is not a dead run
                n_failed += 1
                print(f"  [warn] {item.item_id} failed: {exc}")
                continue
            fh.write(json.dumps(record) + "\n")
            fh.flush()

    print(f"\nWrote {out_path} ({len(items) - n_failed} records, {n_failed} failed).")
    print("Next: python -m src.evaluate --config", args.config)


if __name__ == "__main__":
    main()
