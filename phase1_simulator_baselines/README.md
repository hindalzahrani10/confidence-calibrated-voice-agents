# Phase 1: Closed-loop simulator and baselines (NM-VoI)

> **All findings in this folder are PRELIMINARY.** They come from a synthetic, single-machine feasibility study with a 10-item held-out test split.

**Status: preliminary, synthetic, single-machine feasibility study. Nothing here is a publishable result.**

## Research objective
Test whether a spoken-dialogue agent that accounts for how background noise corrupts the user's *clarification reply*
(Noise-Marginalized Value of Information, **NM-VoI**) decides better when to answer, clarify (open / closed / confirm) or
abstain than (a) simple confidence and entropy thresholds, (b) always/never asking, (c) standard EVPI, and
(d) a value-of-information policy that ignores noise (**noise-independent VoI**). The math is in `notes/Formulation.md`.
The central comparison is NM-VoI vs noise-independent VoI.

## What was built and completed
| Milestone | State |
|---|---|
| 50-item synthetic dataset (`phase1.v2`), validator, 38 negative tests, immutable snapshot | done; annotations are AI-drafted and **not human-reviewed** |
| Audio: Piper TTS + real cafeteria noise at clean / 20 / 10 / 5 dB, 1,132 clips, validator + 18 negative tests | done |
| ASR (faster-whisper small), lexical task-belief model, calibration, blind SNR estimator, channel estimation | done |
| Closed-loop simulator, 7 policies + 6 ablations, tuning on val only, protocol freeze, single held-out run | done |
| Methodological audit of the implementation | done (`limitations_and_next_steps.md`) |
| Manual review of 20 dialogues (D01-D20) by the project author | done (`human_evaluation.md`) |
| Human review of the 50 Phase 1 *dataset annotations* | **not done (deferred)** |
| Paper update | **not started; awaiting approval** |

## Summary of preliminary findings
1. Value-of-information policies with abstention (EVPI, noise-independent VoI, NM-VoI) reached the highest dev utility (about 0.655-0.659) and cut wrong answers to about 3%, versus 12-15% for always-ask and the thresholds. Against the thresholds and always-ask the utility intervals include zero on dev; against never-ask they do not.
2. **NM-VoI did not beat noise-independent VoI.** Dev utility difference -0.004 [-0.011, 0.000]; held-out test (10 items) +0.007 [0.000, 0.022]. The two chose different first actions in only 5 of 160 dev clips and never differed in the final answer.
3. The estimated reply channel is essentially flat across SNR (all clean-vs-5 dB Fisher p >= 0.41) and 5 dB speech degraded only mildly (WER 3.0% -> 9.0%). The acoustic conditions probably do not separate the two policies.
4. Human inspection of 20 dialogues found the main failure sources to be reply-parser failures, belief-model ties, and question selection or wording, not noise in the reply channel.

## Read this before citing anything
- All data are synthetic (hand-authored text, TTS speech, one real noise recording). Annotations are unreviewed.
- The agent's belief is a **lexical word-matching model, not an LLM**; the 7-8B model in `configs/default.yaml` was never run (2 vCPU, 3.8 GB RAM, no GPU).
- The task catalogue always contains the true task (shared by all policies).
- Dev results pool train+val items; the channel table, belief temperature and SNR rule were fitted on the 30 train items, so train-item results are in-sample. Only 10 val items and 10 test items are held out.
- The test split has 10 items (40 episodes per policy): **preliminary**.

## Contents
| File | Purpose |
|---|---|
| [methodology.md](methodology.md) | simulator architecture, setup, noise conditions, policies |
| [results.md](results.md) | dev and test numbers, baseline comparisons, ablations, NM-VoI limits |
| [human_evaluation.md](human_evaluation.md) | analysis of the 20 manually reviewed dialogues |
| [limitations_and_next_steps.md](limitations_and_next_steps.md) | audit findings, remaining limitations, recommended next steps |

**Repository note.** These five documents are self-contained. The code, experiment reports, audit files and generated data they summarise (source modules, run scripts, per-run result tables, audio, ASR outputs) live in the author's working copy and are **not yet in this repository**; nothing here links to them. The mathematical formulation is in [`notes/Formulation.md`](../notes/Formulation.md).

## Open items
Five human-evaluation questions are **pending clarification** (D01, D10, D16 vs D17, D03, D19); see [human_evaluation.md](human_evaluation.md#pending-clarification-five-unresolved-human-evaluation-questions).
