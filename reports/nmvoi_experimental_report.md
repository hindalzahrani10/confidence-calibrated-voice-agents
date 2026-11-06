# NM-VoI experimental report (feasibility study, PRELIMINARY)

**Status of the evidence.** All data are SYNTHETIC: 50 hand-authored items, Piper TTS speech, one real cafeteria noise recording mixed at 4 SNR levels.
The Phase 1 semantic annotations are AI-drafted and have NOT been human-reviewed (review deferred). The held-out test split has only
**10 base items** (40 episodes per policy), so every test result below is **preliminary**. No paid service or new cloud resource was used. Nothing is committed.

## 1. Setup in one paragraph
Actions: answer / open / closed / confirm / abstain. Utility = reward (+1 right, -1 wrong, 0 abstain) minus fixed a-priori costs (open .30, closed .20, confirm .10).
NM-VoI picks the action with the highest one-step expected utility, where the clarification-reply channel r_q(SNR) is estimated from TRAIN replies per question type and per
blindly estimated SNR bucket (the agent never reads the true SNR). Splits: 30 train / 10 val / 10 test items; thresholds (tau=0.4, h=0.75) tuned on val only; protocol frozen
(`reports/protocol_freeze.json`, verified intact after the single test run) before the test split was touched.

## 2. Deviations forced by the host (2 vCPU, 3.8 GB, no GPU)
- No LLM is possible, so the agent's task belief is a transparent lexical model over a 185-task catalogue, softmax temperature fitted on train (beta=2.20). The catalogue always contains the gold task (label-space leakage shared by ALL policies). The results therefore speak to the clarification decision layer, not language understanding.
- ASR = faster-whisper small (int8). One TTS voice, one noise recording (split into disjoint train/val/test time ranges), one noise replicate per clip.
- The blind SNR estimator (99.3% train / 98.2% val accuracy) exploits the speech-free lead-in/tail of the synthetic clips; real recordings would not guarantee that.

## 3. Phase 3 findings (train/val only)
- WER on train by SNR: clean 3.0%, 20 dB 3.7%, 10 dB 5.2%, 5 dB 9.0%. Noise degrades TTS speech only mildly.
- Task inference joint accuracy: 0.63 train, 0.50 val (after fixing a tie-break artefact that had flattered val: 0.70 -> 0.50). By category (train): none 1.0, acoustic 0.88, intent 0.36, underspecified 0.21.
- Calibration (belief temperature fitted on train): ECE 0.125 / NLL 1.07 train; val NLL 0.96 vs 1.56 uncalibrated (beta=1).

## 4. Channel (train replies; recognised-as-intended rate, clean/20/10/5 dB)
confirm-yes 1.00 all; confirm-no .92/.97/.97/.95; closed .95/.95/.93/.89; open .90/.94/.82/.79. Only open answers degrade noticeably. **The channel is nearly noise-flat**, which limits what a noise-marginalised policy can gain.

## 5. Main results

### Dev (train+val, 40 items x 4 SNR, 160 episodes per policy; 95% item-clustered bootstrap)
| policy | success | wrong | abstain | ask rate | unnecessary clar. | utility |
|---|---|---|---|---|---|---|
| never_ask | .600 | .400 | 0 | 0 | 0 | .200 [-.06,.46] |
| always_ask | .875 | .125 | 0 | 1.00 | .600 | .550 [.40,.69] |
| confidence_threshold | .850 | .150 | 0 | .49 | .131 | .602 [.41,.77] |
| entropy_threshold | .850 | .150 | 0 | .53 | .169 | .594 [.40,.76] |
| evpi | .825 | .031 | .144 | .69 | .350 | .657 [.51,.78] |
| noise_indep_voi | .806 | .025 | .169 | .65 | .312 | .659 [.51,.78] |
| **nm_voi** | .806 | .025 | .169 | .68 | .338 | .655 [.51,.78] |

Paired nm_voi minus other (utility): never_ask +0.455 [.23,.70]; always_ask +0.105 [-.04,.23]; confidence_threshold +0.052 [-.04,.16]; entropy_threshold +0.061 [-.03,.17];
evpi -0.002 [-.06,.04]; **noise_indep_voi -0.004 [-.011,.000]**.

### Held-out test (10 items, 40 episodes per policy) -- PRELIMINARY
| policy | success | wrong | abstain | ask rate | unnecessary clar. | utility |
|---|---|---|---|---|---|---|
| never_ask | .500 | .500 | 0 | 0 | 0 | .000 [-.50,.55] |
| always_ask | .775 | .225 | 0 | 1.00 | .500 | .350 [-.15,.80] |
| confidence_threshold | .775 | .225 | 0 | .475 | .075 | .455 [-.03,.88] |
| entropy_threshold | .775 | .225 | 0 | .65 | .175 | .420 [-.09,.85] |
| evpi | .800 | .025 | .175 | .75 | .350 | .635 [.43,.81] |
| noise_indep_voi | .800 | .025 | .175 | .75 | .350 | .645 [.44,.82] |
| **nm_voi** | .800 | .025 | .175 | .725 | .350 | .653 [.45,.82] |

nm_voi vs never_ask +0.653 [.25,1.07]; vs always_ask +0.302 [.05,.63]. Paired nm_voi minus noise_indep_voi on test: +0.007 [0.000, 0.022] (10 items): at most a tiny, lower-bound-at-zero difference, driven by a few ask-rate decisions at 5 dB; not evidence of a meaningful gain.

Burden accompanies success in every table above (ask rate, unnecessary-clarification rate). The VoI family does NOT win on raw success; it wins on utility by cutting wrong answers from 12-15% to ~3% via abstention (abstain 15-17%) at the price of more clarification and more unnecessary clarification (.31-.35 vs .13-.17 for the thresholds). Whether that trade is wanted depends on the reward for abstaining, which was set at 0 a priori, not tuned.

## 6. Ablations (dev, utility; nm_voi .655)
open-only .491, closed-only .592, confirm-only .589, equal-costs .613, uncalibrated belief .596, no-abstain .654. Having all three question types beats any single type (open-only paired +0.164 [.05,.30]); using a calibrated belief helps (+0.059, interval includes 0); equal costs -0.042 (interval includes 0). Removing abstention leaves utility unchanged but raises wrong answers (.106 vs .025) because the reward asymmetry is mild at these costs.

## 7. Does noise-awareness matter? (the actual NM-VoI claim)
**On this evidence, no.** NM-VoI and the noise-independent VoI are statistically indistinguishable on dev (-0.004 [-.011,.000]) and on test (+.007 [0.000, 0.022], 10 items). They differ only in a few ask-rate decisions at 20 and 5 dB. Reason: the channel table is almost flat across SNR for confirm/closed replies and TTS speech is barely hurt at 5 dB. Because the baseline VoI uses the same pooled channel estimate, the data cannot show the benefit of the SNR conditioning. Standard EVPI (assuming a perfect channel) also ties.
What the data do support is the broader claim that **a calibrated expected-utility clarification policy (with abstention) beats never-asking decisively and beats always-asking and score-threshold policies on utility on dev**, mostly through fewer wrong answers.

## 8. Error analysis (by category, dev utility)
- none-ambiguity: VoI policies ~.97 (only the unneeded asks cost them).
- acoustic: noise_indep_voi .76 vs nm_voi .74 vs always_ask .60 / confidence_threshold .77: no gain from noise awareness.
- intent: nm_voi .47 vs always_ask .65 (asking pays off here; lexical belief cannot tell intents apart so VoI abstains too often).
- underspecified: nm_voi .38; no_abstain .47 - abstention costs recoverable cases when replies would have resolved them. Categories contain 9-11 items each; descriptive only.

## 9. Threats to validity
Synthetic text and TTS speech; catalogue contains gold; AI-drafted unreviewed annotations; user simulator is scripted (reply-style robustness to bare vs corrective "no" was NOT swept); one voice, one noise recording, one noise draw per clip; belief is lexical, not an LLM; tiny test split (10 items); costs and reward fixed rather than studied (cost-scale sensitivity not run); thresholds tuned on 10 val items.

## 10. Not done (honest list)
Cost-scale / reply-style / noise-seed robustness sweeps, a second voice or noise type, accent shift, white noise, and any human validation. These were skipped to protect the sealed test split and the compute budget; nothing was substituted by guesses.

## 11. Reproducibility
`reports/REPRODUCE.md`; deterministic seeds; frozen protocol hash; episodes in `data/phase5/{dev,test}.v1/`; tables and figures in `reports/results/{dev,test}/`. Leakage guard: `scripts/test_nmvoi.py` (AST scan, analytic tests) passes.

## 12. Conclusion
Does the evidence support the NM-VoI hypothesis? **Only in part.** It supports value-of-information-driven clarification with abstention over never-ask, always-ask and confidence thresholds (utility, wrong-answer rate), on dev with intervals that mostly include small effects against thresholds and on a 10-item preliminary test. It does **not** support the specific claim that marginalising over noise adds anything beyond a pooled-channel VoI: the paired difference is ~0.
A publication-quality claim needs: real or far more varied speech (multiple voices, accents, real recordings) where the channel actually degrades with SNR; an LLM-based or learned belief with a catalogue that does not contain the answer; a few hundred human-reviewed items with a test split large enough for narrow intervals; multiple noise types and seeds; a real user study or validated simulator; and sweeps over costs and reward for abstention.
