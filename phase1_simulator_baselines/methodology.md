# Methodology

> **Preliminary.** Describes the Phase 1 feasibility implementation as built; all data are synthetic. Implementation files and run commands are in the author's working copy (not yet in this repository). Limitations are in [limitations_and_next_steps.md](limitations_and_next_steps.md).

## 1. Decision problem
Actions: **answer**, **abstain**, or ask one question of type **open**, **closed** or **confirm**.
Utility = reward (+1 correct, -1 wrong, 0 abstain) minus interaction cost, with costs fixed a priori:
open 0.30, closed 0.20, confirm 0.10 (never tuned). "Correct" means the answered task is in the item's `acceptable_tasks` set
(some items accept wildcards, e.g. any music genre). Formulation: [`notes/Formulation.md`](../notes/Formulation.md).

**One-step lookahead.** The agent may ask at most one question, then answers or abstains
(policy module docstring; Formulation section 4.2). A failed or "none of those" reply cannot trigger a second question.

- EU(answer) = 2 max b0 - 1; EU(ask q) = sum over outcomes of p(o) max(0, 2 max b1(.|o) - 1) - C(q).
- The belief is updated by Bayes with the same reply-channel model used for planning.

## 2. Closed-loop architecture
```
 agent side (sees observations only)               evaluator side (holds the hidden task)
 ASR words of initial clip  ---------------->  user simulator  (question-conditioned)
 blind SNR bucket (from samples)                   |  reply text
 lexical belief b0 -> policy -> question  ---->    TTS (Piper) -> noise at the clip's SNR -> ASR (faster-whisper)
 parse reply -> Bayes update -> answer/abstain <---  reply transcript + word probabilities
                                               utility scored against acceptable_tasks
```
- The agent never receives gold labels; an automated AST scan checks agent modules for dataset-private identifiers.
- **Simulator (evaluator side).** Open question -> the Phase 1 full-coverage reply (or "I'm not sure what you mean" if nothing was unresolved). Closed -> the offered option equal to the hidden value, else "None of those." Confirm -> "Yes." if the proposal is acceptable, otherwise "No." or a corrective "No, X" (seeded 50/50 per item and proposal).
- **Replies are spoken, noised and recognised**, never passed as clean text: reply clips come from the Phase 2 inventory or are synthesised on demand with the same code and seeds.

## 3. Agent components
- **Belief model.** Lexical scoring of 185 catalogue tasks against the ASR transcript, weighted by word probability, with a softmax temperature (beta = 2.20) fitted on train only. Exact ties are broken by a label-free hash. **This is not a language model.**
- **Blind SNR estimator.** Noise floor from the recording's lead-in and tail, speech frames 6 dB above it, nearest centroid fitted on train clips. Accuracy 99.3% train, 98.2% val; it relies on the speech-free padding of the synthetic recordings.
- **Reply channel.** Recognition rate r_q(SNR bucket) per reply type (confirm-yes, confirm-no, closed, open), estimated on **train replies only**, with Beta shrinkage toward the pooled rate (strength 4). Outcome model: the intended outcome with probability r, the other outcomes uniformly.

## 4. Data and noise conditions
- 50 items (13 none, 13 acoustic, 12 intent, 12 underspecified), split 30 train / 10 val / 10 test; each item has four variants: clean, 20, 10, 5 dB SNR.
- Speech: Piper `en_US-lessac-medium`, one voice, deterministic synthesis, 16 kHz, active-speech level -26 dB.
- Noise: one real DEMAND PCAFETER recording (CC BY 4.0), split into disjoint time ranges for train [0,180] s, val [180,240] s, test [240,300] s. SNR is defined on active-speech power; achieved SNR within +/-0.01 dB.
- ASR: faster-whisper small, int8, CPU, beam 5, temperature 0. One noise draw per clip.
- Phase 2 validation: 1,132 clips (200 initial + 932 reply), no clipping, determinism re-check 40/40.

## 5. Policies
| Policy | Rule |
|---|---|
| never_ask / always_ask | answer immediately / always ask one closed question on the max-entropy variable |
| confidence_threshold, entropy_threshold | closed question when max prob < tau (0.4) or entropy > h (0.75); tuned on val only; never abstain |
| evpi | VoI with a perfect-channel assumption |
| noise_indep_voi | VoI with the reply channel pooled over SNR |
| **nm_voi** | VoI with the SNR-bucket channel |
| ablations | equal costs; open-only; closed-only; confirm-only; no abstention; uncalibrated belief (beta = 1) |

evpi, noise_indep_voi and nm_voi differ **only** in the channel assumption. Threshold baselines and always_ask update at face value after a reply and cannot abstain or use open/confirm questions.

## 6. Evaluation protocol
- Fit only on train; tune thresholds on val only; test sealed until a protocol-freeze file (hashes of code, config and tuned values) was written; one test run; freeze verified intact afterwards.
- Statistics: 95% percentile bootstrap resampling whole base items (the four SNR variants are one cluster); paired differences against NM-VoI.
- Success is always reported next to burden (ask rate, turns, cost, unnecessary-clarification rate).
