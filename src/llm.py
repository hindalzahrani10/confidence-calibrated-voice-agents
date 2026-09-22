"""The LLM half of the cascade.

Four things are asked of the model per utterance:
  1. a greedy answer, with token log-probabilities  -> token-level signal
  2. k sampled answers                              -> semantic entropy
  3. a verbalized confidence 0-100                  -> verbalized signal
  4. a yes/no "should I clarify?" probe             -> self-interrogation

Prompts live in `prompts/` as plain text so they can be edited without
touching code, which matters because prompt wording is itself an ablation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np


@dataclass
class Generation:
    """One model output plus its token-level score."""

    text: str
    mean_logprob: float = 0.0

    @property
    def perplexity(self) -> float:
        return float(np.exp(-self.mean_logprob))


def load_prompt(prompts_dir: Path, name: str) -> str:
    """Read `prompts/<name>.txt`."""
    return (Path(prompts_dir) / f"{name}.txt").read_text(encoding="utf-8").strip()


class LLMAgent:
    """An open instruction-tuned chat model used as the voice agent."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        device_map: str = "auto",
        dtype: str = "bfloat16",
        max_new_tokens: int = 96,
        prompts_dir: str | Path = "prompts",
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.max_new_tokens = max_new_tokens
        self.prompts_dir = Path(prompts_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            dtype=getattr(torch, dtype),
        )
        self.model.eval()

    def _chat(
        self,
        system: str,
        user: str,
        temperature: float = 0.0,
        max_new_tokens: Optional[int] = None,
    ) -> Generation:
        """Run one chat turn and return the text with its mean log-probability."""
        import torch

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens or self.max_new_tokens,
                do_sample=temperature > 0.0,
                temperature=temperature if temperature > 0.0 else None,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = out.sequences[0, inputs["input_ids"].shape[1] :]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # transition_scores gives the log-probability actually assigned to
        # each token that was emitted, which is what we want regardless of
        # whether we sampled or decoded greedily.
        scores = self.model.compute_transition_scores(
            out.sequences, out.scores, normalize_logits=True
        )[0]
        finite = scores[torch.isfinite(scores)]
        mean_logprob = float(finite.mean()) if finite.numel() > 0 else 0.0

        return Generation(text=text, mean_logprob=mean_logprob)

    def answer(self, query: str) -> Generation:
        """Greedy answer: the response the agent would actually give."""
        return self._chat(load_prompt(self.prompts_dir, "answer"), query)

    def sample_answers(
        self, query: str, k: int = 5, temperature: float = 0.7
    ) -> List[Generation]:
        """k sampled answers, used to estimate semantic entropy."""
        system = load_prompt(self.prompts_dir, "answer")
        return [
            self._chat(system, query, temperature=temperature) for _ in range(k)
        ]

    def verbalized_confidence(self, query: str, answer: str) -> float:
        """Ask the model to rate its own confidence 0-100, return it in [0, 1]."""
        system = load_prompt(self.prompts_dir, "verbalized_conf")
        user = f"User request: {query}\nYour answer: {answer}"
        gen = self._chat(system, user, max_new_tokens=12)

        match = re.search(r"\d{1,3}", gen.text)
        if not match:
            # A model that will not produce a number is maximally unhelpful
            # here; 0.5 records "no usable signal" without faking certainty.
            return 0.5
        return min(max(int(match.group()) / 100.0, 0.0), 1.0)

    def self_interrogate(self, query: str) -> tuple[bool, str]:
        """Ask the model directly whether it needs to clarify.

        Returns (should_clarify, stated_reason).
        """
        system = load_prompt(self.prompts_dir, "self_ask")
        gen = self._chat(system, f"User request: {query}", max_new_tokens=64)

        first_line = gen.text.strip().splitlines()[0] if gen.text.strip() else ""
        should_clarify = first_line.strip().lower().startswith("yes")
        return should_clarify, gen.text.strip()

    def clarifying_question(self, query: str) -> str:
        """Generate exactly one clarifying question."""
        system = load_prompt(self.prompts_dir, "clarify")
        return self._chat(system, f"User request: {query}", max_new_tokens=48).text

    def answer_with_clarification(
        self, query: str, question: str, user_reply: str
    ) -> Generation:
        """Answer after the simulated user has replied to our question."""
        system = load_prompt(self.prompts_dir, "answer")
        user = (
            f"User request: {query}\n"
            f"You asked: {question}\n"
            f"User replied: {user_reply}\n"
            "Now give the final answer."
        )
        return self._chat(system, user)
