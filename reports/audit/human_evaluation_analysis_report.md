# Human evaluation analysis of 20 closed-loop dialogues (D01–D20)

Status: **for your review; nothing here has been added to the paper.** All data are synthetic (hand-authored text, Piper TTS, DEMAND cafeteria noise). The dialogues are **dev episodes (train/val items)**, selected for coverage and for interest, not sampled at random; the held-out test split was not used. Frozen experimental results were not modified; the original CSV, original Excel workbook and the merged CSV were checksum-verified unchanged after the analysis.

Labels used below: **[V]** verified from logs/files/your entries, **[I]** my interpretation, **[H]** hypothesis not tested.

## 1. Data handling

- The Excel workbook is treated as the only source of human judgments. The original CSV's `H_` columns were all blank and `H_reviewed_by_human` was `NO` in every row, so no stale CSV values entered the merge **[V]**.
- **D19 question rating** `Y` → `5` (your confirmation). This is the only standardisation; raw and normalised values are both kept in `review_human_normalized.csv` with a note.
- **D01 is flagged, not resolved.** In Excel: audio/ASR = `N`, final action correct = `unsure`, overall valid = `unsure`; your note says "whether clarification was preferable remains unresolved." No assumption was made. In counts below D01 is reported as `unsure`, and it is not used in any correct/incorrect denominator **[V]**.
- Files created (all new): `review_package/review_human_normalized.csv` (human raw + normalised + automatic evidence + `A_` analyst codings, clearly labelled), `review_package/human_review_analysis.json`, `scripts/analyze_human_review.py`, and this report.

## 2. Descriptive statistics (n = 20; 13 with a clarification, 7 without) **[V]**

| Measure | Result |
|---|---|
| Blank question/reply evaluations | exactly the 7 dialogues without a clarification (D01, D07, D08, D10, D11, D14, D18); all 13 asking dialogues are filled |
| Question rating (1–5, n = 13) | mean 3.85, median 5, IQR 3–5; counts: 5×7, 4×1, 3×2, 2×2, 1×1 |
| …by question type | confirm n=4: all 5 (mean 5.0); closed n=5: mean 3.6 (ratings 5,5,5,2,1); open n=4: mean 3.0 (ratings 4,3,3,2) |
| …by policy | always_ask n=3: 2.67 (5,2,1); confidence_threshold n=2: 5.0; noise_indep_voi n=2: 5.0; nm_voi n=3: 3.67 (5,4,2); evpi n=3: 3.67 (5,3,3) |
| …by final outcome | success n=9: mean 4.56; wrong n=1: 1; abstain n=3: 2.67 |
| Reply appropriateness (n = 13) | Y 12, N 1 (D03) |
| Audio/ASR consistency (n = 20) | Y 16, N 3 (D01, D03, D08), unsure 1 (D04) |
| Final action correct (n = 20) | Y 14, N 4 (D08, D17, D18, D20), unsure 2 (D01, D03) |
| Overall valid (n = 20) | Y 17, unsure 3 (D01, D08, D19) |

With 20 hand-picked dialogues, 13 asking and ≤5 per question-type or policy cell, these numbers are descriptive. No cell supports a significance claim, and the sample cannot estimate how often any failure occurs in the full dev set **[I]**.

## 3. Human vs automated assessments

**Agreement [V]**
- **Transcript accuracy.** The automatic flags "ASR differs from reference/intended" (F1/F8) are on exactly D01, D03, D04, D08, and exactly those are the four dialogues you did not mark `Y` for audio/ASR consistency (20/20 dialogues agree).
- **Answered dialogues.** For the 14 dialogues where the agent answered, your "final action correct" matches the automatic success/wrong outcome in 13/14 (93%; Wilson 95% interval 69–99%).

**Disagreements and ambiguities [V unless stated]**

| ID | Human | Automatic | Comment |
|---|---|---|---|
| D10 | final correct = Y | wrong answer (−1.0; "book a flight" vs gold hotel) | Your note says the tie "led to the wrong choice." Likely a labelling slip or a different construct (policy-consistent?). **Please confirm.** |
| D16 vs D17 | D16 final = Y, D17 final = N | both abstained after an unusable reply | Same situation, opposite labels. D17's initial answer ("play some rock") would have been acceptable (wildcard genre), D16's ("8 am") was not **[V]**, which may explain it **[I]**. For abstentions the construct "final action correct" is not defined; across the 6 abstentions: Y 1, N 3, unsure 2. |
| D03 | reply appropriateness = N | simulator reply "Dan" is the intended answer; ASR gave "Then" | Your note says you heard "Dan." N appears to reflect the recognised reply, not the reply itself **[I]**. Please confirm the intended meaning of that column. |
| D19 | audio/ASR = Y | no flag | Your note says you could not identify the utterance ("turn off the vorm"?) at 5 dB while ASR gave "Turn up the volume." Possibly Y means "ASR matched the reference." **Please confirm.** |
| D04 | audio/ASR = unsure | F1 (reply ASR "Bye, flower." vs "Buy flour") | You also heard the *original* as "buy a flower"; ASR and reference say "flour". This item is a designed homophone; the TTS audio itself is ambiguous to a human listener **[V]**. |
| D17, D15, D02, D04, D05, D13, D12 | notes on whether asking was needed | F10 "unnecessary clarification" on D02, D03, D04, D05, D12, D13, D15, D17 | See §6: F10 is defined by the initial MAP task being acceptable, which does not match your judgments of necessity in several cases. |

## 4. Failure taxonomy (analyst coding of each dialogue; evidence in the CSV)

| Class | Dialogues | Evidence **[V]** |
|---|---|---|
| **Acoustic / ASR error on the initial utterance** | D01 (10 dB, "Come on."), D03 (20 dB, "Cole Dan."), D08 (5 dB, "Place Reelers.") | Your N ratings; transcripts in the logs. D01 and D08 have the gold task outside the top-5 beliefs. |
| **ASR error on the clarification reply** | D03 ("Dan" → "Then", 20 dB), D04 ("Buy flour" → "Bye, flower.", clean) | The D04 error occurs at **clean** SNR (a homophone), so it is not noise-driven. |
| **Reply-parser / understanding failure (reply transcribed correctly)** | D16 ("7 in the morning." → gold task scores −0.658; grammar mismatch), D17 ("Jazz." exact, but ASR word probability 0.163 gives score 0.326 below the hard 0.5 gate) | I re-scored the logged transcripts with the existing belief model (read-only, no experiment re-run). You independently called both "parsing failure, not ASR failure." D04's closed reply is also unparseable ("Bye, flower." scores below the 0.6 acceptance). |
| **Clarification-policy: question choice / wording** | D20 (always_ask asked an *intent* question, "book ticket / weather / reminder," though the missing information was the event); open wording generic in D03, D06, D16, D17 (ratings 2, 4, 3, 3) | Question text in the logs; your ratings and notes. D20's variable choice is the baseline's max-entropy rule (`_best_var`), not VoI. |
| **Clarification-policy: asking when not needed** | D02 (max-prob .945, EU margin .0035), D05 (.872), D12 (.9973, always_ask by design), D13 (.653, debatable) | Logs; your notes. |
| **Clarification-policy: not asking when you would have** | D08 (all ask EUs negative: ask_open −0.30), D18 (ask_open −0.2044 vs abstain 0), D01 (unresolved) | EUs logged. Under the one-step model abstaining is the maximiser, so this is a model-adequacy question rather than a bug **[I]**. |
| **Belief-model artefact** | D15 ("9 p.m." transcribed correctly, belief split .4998/.4998 between 9 pm and 9 am) | Logged beliefs; your note attributes it to the belief model, consistent with my earlier audit. |
| **Genuine intent/underspecification tie** | D07 (calendar/reminder/note, 1/3 each, correct by hash tie-break), D10 (hotel/flight/car, wrong), D19 | Logged beliefs. D07's "success" is luck among three equals **[V]**. |
| **Downstream execution** | D20: after the reply fixed the intent, the event was a 3-way tie and the final pick (comedy show) was wrong (−1.2) | Logged updated belief (.3333 each). It is a consequence of the earlier question choice, not an independent error. |
| **No problem found** | D06, D09, D11, D14, D19 | Your Y ratings and logs. |

## 5. Clarification quality separately from final success

Outcome of the 13 asking dialogues, from logged initial vs final correctness **[V]**:

| Effect of the clarification | n | Dialogues |
|---|---|---|
| Rescued (initial MAP wrong → final right) | 3 | D06, D09, D19 |
| Confirmed (initial MAP already right → right) | 5 | D02, D05, D12, D13, D15 |
| Reply unusable, success came from the prior | 1 | D04 |
| Reply unusable, abstained although initial MAP was right (value lost) | 2 | D03, D17 |
| Reply unusable, abstained, initial MAP wrong | 1 | D16 |
| Failed: wrong final answer | 1 | D20 |

- Only 3/13 asks demonstrably changed a wrong answer into a right one (Wilson 95% interval ≈ 8–50%); the other 5 "successes" were already right before asking, and D04's success is not attributable to the clarification at all because its reply was never used (belief unchanged at .9944).
- 4/13 replies (31%, interval ≈ 13–58%) were unusable even though the intended reply was spoken. Two (D16, D17) were transcribed correctly, so this loss is parser-side, not acoustic.
- **Question quality and outcome are related but separable [I]:** higher ratings go with successes (mean 4.56, n=9) than with abstains/wrong (2.67 and 1.0, n=4), but D12 (rating 2) still ended in success and D04 (rating 5, relevant question) ended in a parser failure. A good question can fail downstream and a poor one can succeed.
- Final success is also not a clean measure of agent quality: D07 succeeded by tie-break luck and D10 failed by tie-break luck.

## 6. The "unnecessary clarification" metric (F10) **[V]** and **[I]**

F10 marks an ask as unnecessary if the initial MAP task is already acceptable. Problems visible in this sample:
- D17: the item is "low-stakes" (any genre is acceptable; Phase 1 `acceptable_tasks` wildcard), so the arbitrary MAP choice "rock" counts as acceptable and the ask counts as unnecessary, while you judged the ask appropriate (no genre was stated).
- D15: the MAP is "correct" only because of the hash tie-break between 9 pm and 9 am; you called the ask reasonable.
- Across the full dev set, **25 of the 54** nm_voi episodes marked unnecessary had an initial near-tie (computed from logs), so a large share of the headline unnecessary-clarification rate (0.34) depends on tie-breaking luck.
- Conclusion **[I]:** the metric measures "ask when the arbitrary top choice was right," not "ask when the user was clear." It should not be reported as user-perceived burden.

## 7. Limitations

- n = 20, non-random (coverage-driven plus flagged cases), mixed policies (6) and conditions; no prevalence estimates, no significance tests.
- **Single rater, not blinded:** the review package showed the reference text, the agent's ASR and the agent's decisions next to the audio, so ratings of ASR consistency and decisions may be anchored (D08 note: you could not judge independent of the reference). No inter-rater reliability.
- The rating scale conflates three things (relevance, necessity, wording): D12 (2) is mostly necessity, D16/D17 (3) mostly wording, D20 (1) relevance. "Final action correct" is undefined for abstentions; "reply appropriateness" was used for one ASR-failure case.
- Dev episodes only; train items are in-sample for the belief temperature, channel table and SNR centroids.
- Synthetic TTS (you noted a robotic voice in D05), one noise recording, lexical belief rather than an LLM, unreviewed AI-drafted annotations.
- My taxonomy and the diagnostics of D16/D17 are analyst codings of logged data; they were not verified by a second person.

## 8. Implications for NM-VoI (interpretations, not conclusions)

1. **[V]** None of the 20 dialogues shows a failure attributable to the noise-dependent *reply channel* that NM-VoI models. Problems occurred at clean SNR (D04, D05, D10, D20) as well as at 5–20 dB; reply ASR errors were a clean-condition homophone (D04) and one 20 dB case (D03).
2. **[I]** The dominant, noise-independent sources are the reply parser (D04, D16, D17), belief ties/lexical artefacts (D07, D10, D15), and question selection/wording (D03, D06, D16, D17, D20). These can swamp any SNR effect in the channel, consistent with the null NM-VoI-vs-noise-independent result.
3. **[I]** Acoustic damage in this sample hit the *initial* utterance (D01, D03, D08), i.e. the prior belief, not the clarification channel. The current formulation marginalises noise only over the reply; the prior is a single temperature fitted across all SNRs. **[H]** a noise-conditioned prior (or ASR-confidence-aware belief) may matter more than the reply channel.
4. **[H]** The abstentions in D08/D18 may reflect the top-8 support used in the open-question outcome model, which could cap the modelled value of asking when belief is diffuse. The EUs are logged, but no counterfactual was run.
5. **[V]** Reported success and utility mix clarification quality with luck and parser failures, so they cannot by themselves show that calibrated clarification works. D07, D04, D12 successes were not due to useful clarification.
6. **[I]** Because the channel table counts parser failures as "not recognised," measured recognition rates are confounded with parser/grammar errors (D16, D17, D04), which also limits the interpretation of the flat SNR dependence.

## 9. Recommendations (nothing implemented; awaiting your approval)

1. **Confirm the open points:** D01 (final correct / overall valid), D10 (`Y` vs wrong answer), D16 vs D17 abstention labels, D03 reply appropriateness, D19 audio/ASR meaning. I will record your answers in a separate corrections file rather than editing the workbook.
2. **Do not put the 20-dialogue counts in the paper as rates.** At most, cite them as qualitative error analysis, labelled as non-random, single-rater, synthetic.
3. **Analysis-only (no re-run) next step:** on the logged dev episodes, decompose all 108 asks into reply-unusable causes (ASR error vs grammar vs word-probability gate) and report "usable-reply rate" and "rescued rate" beside utility, and recompute the unnecessary-clarification rate excluding near-tie episodes.
4. **If you want parser or belief fixes,** they should be a new, pre-registered version (v2) applied to all policies, evaluated first on dev and then a fresh frozen test run, with v1 results kept; no tuning on the current test items.
5. **For any future human review:** blind listening first (hide ASR and reference), a second rater on a subset, separate columns for relevance/necessity/wording, a defined rubric for abstentions.
6. **Manuscript wording** supportable now: the pipeline is a controlled synthetic testbed; VoI-with-abstention reduces wrong answers vs. never-ask/always-ask in dev; noise-aware vs noise-independent showed no difference; human inspection found the main failure sources to be parser, belief and question-selection issues. Not supportable now: any claim that NM-VoI's noise-dependence improves decisions, or that clarifications are "calibrated" in an LLM sense.

I have not changed the paper and will wait for your decisions.
