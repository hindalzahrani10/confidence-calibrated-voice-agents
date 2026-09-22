# voice-clarify-calibration-starter

Starter code for **Confidence-Calibrated Voice Agents That Know When to Clarify**.

Hind, this is the scaffold for your 10-week project. It already implements
the parts that are fiddly but not interesting (extracting five families of
confidence signal, computing calibration metrics correctly, keeping the test
split honest) so that your time goes into the parts that are actually your
research: the data, the policies, and the analysis.

## The question this code is built to answer

When a voice agent is uncertain, the uncertainty comes from at least three
different places:

| Source | What happened | What the agent should do |
| --- | --- | --- |
| **Acoustic** | the recognizer misheard you | confirm the word ("did you say *flour*?") |
| **Intent** | the transcript is right, the meaning is not | offer a disambiguating choice |
| **Underspecified** | a required detail is missing | ask for that detail only |

A single confidence number cannot express that difference. This repo lets you
measure whether the available signals *can*, and whether a calibrated policy
built on them beats both always-asking and never-asking.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

You need a GPU for comfort, not for correctness. Whisper-small plus a 7 to 8
billion parameter chat model fits on one consumer GPU. If your GPU is small,
set `llm.model: Qwen/Qwen2.5-1.5B-Instruct` in the config for the pilot and
scale up later: the pipeline is identical.

## First three things to run

```bash
# 1. Build a tiny evaluation set (no TTS needed yet) and check the plumbing
python scripts/build_data.py --config configs/default.yaml --skip-tts

# 2. Pilot the whole pipeline on 20 items
bash scripts/run_pilot.sh

# 3. Score what came out
python -m src.evaluate --config configs/default.yaml
```

Step 2 is the important one. A 20-item pilot catches almost every pipeline
bug, and it costs minutes instead of hours. Do not run the full sweep until
the pilot output looks right.

## Layout

```
configs/default.yaml     every knob for a run; copy it per experiment
prompts/                 the four prompts, as plain text so you can iterate
src/
  config.py              config loading and seeding
  data.py                EvalItem, noise variants, label IO
  asr.py                 Whisper wrapper that returns word-level confidence
  llm.py                 answer, k samples, verbalized confidence, self-probe
  signals.py             the five signal families  <- the heart of the project
  policies.py            never / always / thresholds / self-ask / fusion
  metrics.py             ECE, Brier, AUROC, risk-coverage, burden, bootstrap
  run_agent.py           runs both branches, logs everything to JSONL
  evaluate.py            JSONL -> the paper's two tables
scripts/
  build_data.py          construct and freeze the evaluation set
  run_pilot.sh           20-item smoke test
  run_full.sh            full sweep
experiments/
  log_template.md        one row per run, mapping onto the results table
```

## How the code maps onto the paper

`run_agent.py` deliberately runs **both** branches for every item: it answers
directly, and it also asks a clarifying question and answers again. That is
more compute per item, but it means policy comparison is pure post-processing.
You can invent a sixth policy in week 8 and evaluate it without re-running a
single model.

- `results/policy_comparison.csv` fills **Table 1** of the paper.
- `results/signal_calibration.csv` fills the **ablation grid** (Figure 2b).
- `experiments/log_template.md` rows map one-to-one onto table rows.

## Three rules that protect your results

1. **Never tune on the test split.** Thresholds and the fusion model are fit
   on `val` only. `build_data.py` freezes the split and `evaluate.py` enforces
   it. Breaking this invalidates every headline number in your paper.
2. **Always report burden next to accuracy.** Always-clarify wins on accuracy
   alone, which is exactly why accuracy alone is the wrong headline.
3. **Freeze the data at the end of Week 4.** After that, policies change and
   the data does not.

## Things left for you on purpose

These are marked in the code and are genuinely part of the research, not
chores:

- `src/signals.py::cluster_by_meaning` uses token overlap as a cheap stand-in
  for entailment-based clustering. Swapping in an NLI model is a clean
  ablation for the paper.
- `src/run_agent.py::is_correct` uses lenient string containment. Upgrading to
  an LLM judge is worth doing once the plumbing works; report which scorer
  produced your headline numbers.
- `src/signals.py::attribute_uncertainty_source` is a rule-based baseline for
  Metric 9. Beating it with a learned classifier is a natural extension.
- `scripts/build_data.py` ships only the underspecification set. Adding SLURP
  and voiced AmbigQA / ClarQ-LLM items is your Week 4 task.

## If something breaks

Start with `results/runs.jsonl`: every record carries the transcript, both
answers, all five signals and the latency. Most bugs are visible in the first
three records.
