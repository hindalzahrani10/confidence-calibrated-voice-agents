# Human evaluation of 20 dialogues (PRELIMINARY, single rater)

> Five human ratings are **pending clarification** (listed at the end of this file). Nothing here is a validation of the dataset annotations.

Source: the author's local human-evaluation analysis (ratings workbook, normalized CSV and analysis script; not in this repository).
The project author listened to and rated 20 dev dialogues (D01-D20). The Excel workbook is the authoritative record.
Labels: **[V]** verified from logs or the author's entries; **[I]** interpretation; **[H]** untested hypothesis.

## 1. Sample and method
- 20 dialogues from train/val dev episodes (the test split was not used), picked by greedy coverage of ambiguity category, SNR, question type and policy, plus 3 NM-VoI vs noise-independent disagreements and several automatically flagged inconsistencies. **Not a random sample**: counts are not prevalence estimates.
- 13 dialogues contained a clarification; 7 did not (the question/reply ratings are blank for exactly those 7) **[V]**.
- Ratings: question relevance (1-5), reply appropriateness, audio/ASR consistency, final action correct, overall valid, free-text notes. One rater, not blinded to the reference text, the agent's ASR or its decision; no inter-rater reliability.
- The only standardisation: D19 question rating `Y` -> `5` (confirmed by the author). **D01 is left unresolved** (final action correct and overall valid are `unsure`) and excluded from correct/incorrect counts.

## 2. Descriptive results (n = 20) **[V]**
| Measure | Result |
|---|---|
| Question rating (n = 13) | mean 3.85, median 5, IQR 3-5 (5 x7, 4 x1, 3 x2, 2 x2, 1 x1) |
| By question type | confirm n=4: all 5; closed n=5: mean 3.6; open n=4: mean 3.0 |
| By final outcome | success n=9: mean 4.56; abstain n=3: 2.67; wrong n=1: 1.0 |
| Reply appropriateness (n = 13) | Y 12, N 1 (D03) |
| Audio/ASR consistency | Y 16, N 3 (D01, D03, D08), unsure 1 (D04) |
| Final action correct | Y 14, N 4 (D08, D17, D18, D20), unsure 2 (D01, D03) |
| Overall valid | Y 17, unsure 3 (D01, D08, D19) |

Cells are tiny (2-5 dialogues per policy or question type); no significance claims are made.

## 3. Human vs automated assessments
**Agreement [V].** The automatic "transcript differs" flags (D01, D03, D04, D08) are exactly the four dialogues not rated `Y` for audio/ASR consistency. For the 14 answered dialogues, the author's final-action rating matches the automatic outcome in 13 (93%, Wilson 95% interval 69-99%).

**Disagreements and ambiguities [V].**
- **D10:** rated final action correct `Y`, but the agent answered wrongly (utility -1.0) after a 3-way tie. Needs the author's confirmation.
- **D16 vs D17:** both abstained after an unusable reply; D16 `Y`, D17 `N`. "Final action correct" is not defined for abstentions (across the 6 abstentions: Y 1, N 3, unsure 2).
- **D03:** reply appropriateness `N` although the intended reply was "Dan"; likely reflects the recognised reply "Then".
- **D19:** audio/ASR `Y` although the author could not identify the utterance at 5 dB while ASR was right.
- **D04:** the author heard the original as "buy a flower" while ASR and reference say "flour" (a designed homophone): ambiguous to a human listener as well.
- **"Unnecessary clarification" flag (F10)** disagrees with the author's judgments in several cases (D17: any genre is acceptable, so the arbitrary top choice counts as acceptable, but the author found asking appropriate; D15: correct only through a tie-break).
- Unresolved: D01 (clarification preferable?) - flagged, not assumed.

## 4. Errors identified (analyst coding of logged evidence)
| Class | Dialogues | Evidence |
|---|---|---|
| ASR / acoustic error, initial utterance | D01 (10 dB "Come on."), D03 (20 dB "Cole Dan."), D08 (5 dB "Place Reelers.") | author's `N` ratings; transcripts |
| ASR error on the reply | D03 ("Dan" -> "Then"), D04 ("Buy flour" -> "Bye, flower.", **clean**) | reply transcripts |
| Reply-parser failure with correct transcript | D16 ("7 in the morning": grammar mismatch), D17 ("Jazz.": ASR word probability .163 gives score .326, below the 0.5 gate) | logged transcripts re-scored read-only with the existing model; the author independently called both parsing failures |
| Question choice / wording | D20 (always_ask asked a task-type question when the event was missing); generic open wording in D03, D06, D16, D17 (ratings 2, 4, 3, 3) | question text, ratings |
| Asked when not needed | D02 (.945 confidence, EU margin .0035), D05 (.872), D12 (.9973, always_ask by design), D13 (.653, debatable) | logs, notes |
| Not asked where the author would have | D08, D18 (all ask EUs negative under the model), D01 (unresolved) | logged EUs |
| Belief-model artefact | D15 ("9 p.m." heard correctly, belief .4998/.4998 for 9 pm vs 9 am) | logged belief |
| Genuine ambiguity ties | D07 (right by tie-break luck), D10 (wrong) | logged beliefs |
| Downstream consequence | D20: after the intent was fixed the event was a 3-way tie, final pick wrong (-1.2) | updated belief |

## 5. Clarification quality vs final success **[V]**
Of 13 dialogues with a clarification: **rescued** (initial top choice wrong -> final right) 3 (D06, D09, D19); **confirmed** (already right) 5 (D02, D05, D12, D13, D15); reply unusable 4 (D03, D04, D16, D17), of which D04 still succeeded purely from its prior and D03/D17 abstained although the initial choice was acceptable; failed 1 (D20).
- Only 3/13 clarifications demonstrably changed the outcome for the better (Wilson interval about 8-50%).
- 4/13 replies were unusable although spoken correctly; D16 and D17 were transcribed correctly, so these are parser-side losses.
- Higher question ratings accompany successes (mean 4.56) versus abstains/wrong (mean 2.67 and 1.0), but D12 (rating 2) succeeded and D04 (rating 5) hit a parser failure: quality and outcome are related but separable **[I]**.
- Success is not a clean quality measure: D07 succeeded and D10 failed by tie-break.

## 6. Implications for NM-VoI
- **[V]** No failure in these 20 dialogues is attributable to the noise-dependent reply channel; problems also occur at clean SNR (D04, D05, D10, D20).
- **[I]** The main sources are noise-independent (reply parser, belief ties, question selection), which can swamp any SNR effect.
- **[I]** Acoustic damage hit the initial utterance (D01, D03, D08), i.e. the prior belief rather than the reply channel NM-VoI models.
- **[H]** A noise-conditioned prior, or an ASR-confidence-aware belief, may matter more than the reply channel; the top-8 support in the open-question model may cap the modelled value of asking when belief is diffuse (D08, D18). Neither was tested.

## 7. Limitations of this analysis
Twenty hand-selected dialogues, single unblinded rater, a scale that mixes relevance, necessity and wording, undefined "correct" for abstentions, dev-only data (train items in-sample), synthetic TTS speech (the author noted a robotic voice in D05), and analyst codings not checked by a second person.

## Pending clarification: five unresolved human-evaluation questions
These answers have **not** been provided yet. Until they are, the affected ratings are reported exactly as entered and are not interpreted further.
1. **D01:** what are the intended final-action-correct and overall-valid ratings? (Currently `unsure` / `unsure`; whether clarification was preferable is unresolved.)
2. **D10:** final action was rated correct (`Y`) although the agent answered wrongly (utility -1.0) after a 3-way tie. Is `Y` a slip?
3. **D16 vs D17:** both abstained after an unusable reply but were rated `Y` and `N`. What does "final action correct" mean for an abstention?
4. **D03:** reply appropriateness is `N`, but the note says the reply "Dan" was heard. Does `N` refer to the recognised reply ("Then")?
5. **D19:** audio/ASR consistency is `Y`, but the note says the utterance could not be identified at 5 dB. Does `Y` mean the ASR matched the reference?
