# Limitations and recommended next steps

Sources: the author's local methodological audit and human-evaluation analysis (not in this repository). **All findings are PRELIMINARY.** Nothing below has been implemented;
the next steps are **proposals awaiting approval**, and none should be chosen to obtain better-looking results. A null result for
NM-VoI vs noise-independent VoI is a valid finding.

## 1. Methodological audit: requirement verdicts
| # | Requirement | Verdict |
|---|---|---|
| 1 | Belief update after a noisy reply; utility scored against true intent; no ground truth to the policy | Satisfied (caveats below) |
| 2 | One-step lookahead, explicitly documented | Satisfied in code and formulation; only partly stated in the experimental report |
| 3 | Replies really pass through TTS, noise and ASR | Satisfied (108/108 nm_voi dev asks have an audio file and ASR record; 58 from the Phase 2 inventory, 50 generated on demand) |
| 4 | Channel estimated on training data, by SNR and question type, without val/test leakage | Satisfied for estimation; dev evaluation is partly in-sample |
| 5 | Replies relevant to the question actually asked | Partially satisfied |
| 6 | Baseline fairness | Partially satisfied |
| 7 | Lexical belief distinguished from the intended 7-8B LLM | Satisfied |

Key caveats behind those verdicts:
- **Req 1.** When a reply is unparseable the belief is left unchanged and the cost is still paid (19 of 108 nm_voi dev asks). The corrective-"no" path adds lexical evidence outside the channel model. Constants such as the 8-task support, the 0.5 open-reply gate, 0.6 closed-parse acceptance and shrinkage strength 4 are hand-set in code, not in config; how they were chosen cannot be shown from artifacts.
- **Req 2.** A single question with no re-ask matters here because 17.6% of asks are wasted with no recourse.
- **Req 4.** Dev results pool train and val, so for the 30 train items the channel, belief temperature and SNR centroids are fitted on the scored items. The channel is estimated from scripted Phase 1 probe replies, not from the replies the agent's own questions elicit. The grammar file hash matches the one recorded when the temperature was fitted.
- **Req 5.** Open replies are fragments written for specific Phase 1 probes ("Jazz", "Dan") but are given in answer to a generic open prompt. The "No" branch of confirmation (3 of 33 confirm asks) and "None of those" (1 of 56 closed asks) are barely exercised. The simulated user never mishears or hesitates.
- **Req 6.** Threshold baselines and always_ask can only ask one closed question and cannot abstain, so the VoI-vs-threshold gap mixes decision rule with action space. Thresholds were tuned on 10 val items that are also in the dev pool.
- **Req 7.** "Calibrated" here means one temperature on a word-overlap model, not LLM confidence.

## 2. Additional audit findings
1. **Tied beliefs.** In 82 of 160 NM-VoI dev episodes the top two tasks tie within 1e-3 and a hash breaks the tie. 72 are designed intent or underspecified ambiguity; 8 are "none"-category items that should be unambiguous (e.g. "...saying I'm running late" vs "...saying running late"; "9 p.m." vs "9 am"). The cause is not diagnosed.
2. **Near-indifferent decisions.** Median EU margin between the chosen action and the runner-up is 0.10; 6.9% are within 0.02.
3. **Unusable replies.** 19 of 108 asks (17.6%). Eight are two items that fail at every SNR including clean ("Seven in the morning" is transcribed "7 in the morning." and not parsed), so the measured channel reliability mixes parser failures with acoustic ones.
4. **NM-VoI and noise-independent VoI behave almost identically** (first action differs in 5/160, final outcome never).
5. **Human review findings** (see [human_evaluation.md](human_evaluation.md)): parser failures despite correct transcripts, generic open-question wording, the always_ask baseline choosing the wrong variable, and an "unnecessary clarification" flag that depends on tie-breaks (25 of 54 flagged NM-VoI dev episodes began from a near-tie).

## 3. Remaining limitations affecting the central hypothesis
- **Insufficient acoustic variation.** WER 3.0% -> 9.0% from clean to 5 dB; the estimated channel is flat within sampling error (cell sizes 22-60; only drops of roughly 20+ points would be detectable). One clear TTS voice, one noise recording with artificial speech-free padding, worst case 5 dB, one noise draw per clip, short fixed replies.
- **Belief model.** Lexical, not an LLM; the 7-8B model never ran. The catalogue always contains the true task. Belief ties create artefactual ambiguity.
- **Sample size.** 50 items; 10 test items (40 episodes per policy); 10 val items.
- **Annotations.** Phase 1 semantic annotations are AI-drafted and unreviewed by humans; only 20 dialogues were reviewed, by one unblinded rater.
- **Simulator.** Deterministic scripted user; reply style (bare vs corrective "no") not swept.
- **Not run:** cost-scale and abstention-reward sensitivity, noise-seed robustness, second voice or noise type, accents, white noise.
- **Blind SNR estimator** depends on the speech-free lead-in and tail of the synthetic clips.
- **Initial utterance vs reply.** NM-VoI marginalises noise over the reply only; the prior belief uses a single temperature across all SNRs.

## 4. Recommended next steps (proposals, in priority order)
1. **Resolve the five human-review questions, which are pending clarification** (D01, D10, D16 vs D17, D03, D19; wording in [human_evaluation.md](human_evaluation.md#pending-clarification-five-unresolved-human-evaluation-questions)), recording the answers in a separate corrections file rather than editing the ratings workbook.
2. **Analysis-only on logged dev episodes (no re-run):** decompose the 19 unusable replies by cause (ASR error, grammar mismatch, word-probability gate); report usable-reply rate and "rescued" rate beside utility; recompute unnecessary-clarification excluding near-tie episodes.
3. **Power analysis before more experiments**, then widen acoustic conditions only as far as needed to make the channel measurably change (lower SNR, several noise types, several voices, accents or real recordings, reply replicates). Report channel reliability before looking at policy performance.
4. **Re-estimate the channel from closed-loop replies and score only out-of-sample items** (cross-validation over items) so no item is scored with its own channel.
5. **Make the baselines comparable:** threshold policies with the same action space (abstention, confirm and open questions).
6. **If parser or belief fixes are wanted,** treat them as a pre-registered v2 applied to all policies, evaluated first on dev and then on a fresh frozen test run. Keep v1 results. Do not tune on the current test items.
7. **Pre-stated sensitivity sweeps:** cost scales, abstention reward, reply style, a simulator that sometimes errs.
8. **Human review of the Phase 1 annotations** (deferred) before any claim about validity; for later dialogue reviews use blind listening, a second rater, separate columns for relevance/necessity/wording, and a rubric for abstentions.
9. **An LLM belief model** would test the original research question about confidence signals, but needs hardware not available on this host; it is not proposed as an action.

## 5. What can and cannot be said now
- **Supportable:** the pipeline is a controlled synthetic testbed; VoI-with-abstention lowers wrong answers relative to never-ask, always-ask and thresholds in dev; noise-aware and noise-independent VoI were indistinguishable here; human inspection found parser, belief and question-selection issues as the main failure sources.
- **Not supportable:** that modelling the noise-dependent clarification channel improves decisions; that results transfer to real speech, real users or LLM agents; that the annotations are human-validated.
