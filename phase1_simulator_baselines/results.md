# Results (PRELIMINARY)

**All numbers are PRELIMINARY.** Synthetic data; lexical belief model; see [limitations_and_next_steps.md](limitations_and_next_steps.md). Source: the author's experimental report and per-run result tables (local, not in this repository). Intervals are 95% bootstrap over whole base items. Numbers copied from those files.

## 1. Upstream measurements (train/val only)
- Initial-utterance WER by SNR (train): clean 3.0%, 20 dB 3.7%, 10 dB 5.2%, 5 dB 9.0%. Noise degrades this TTS speech only mildly.
- Task-inference joint accuracy: 0.63 train, 0.50 val (after fixing a tie-break artefact that had inflated val to 0.70). By category (train): none 1.0, acoustic 0.88, intent 0.36, underspecified 0.21.
- Belief calibration (one temperature fitted on train): ECE 0.125 / NLL 1.07 on train; val NLL 0.96 vs 1.56 uncalibrated.
- Reply channel, recognised-as-intended rate (train; clean / 20 / 10 / 5 dB): confirm-yes 1.00 all; confirm-no .92/.97/.97/.95; closed .95/.95/.93/.89; open .90/.94/.82/.79. Clean vs 5 dB Fisher p: 1.00, 0.72, 0.43, 0.41 (no significant SNR dependence).

## 2. Dev comparison (train+val, 40 items x 4 SNR = 160 episodes per policy)
Note: 30 of the 40 items were used to fit the belief temperature and channel table (in-sample); only the 10 val items are held out.

| policy | success | wrong | abstain | ask rate | unnecessary clar. | utility |
|---|---|---|---|---|---|---|
| never_ask | .600 | .400 | 0 | 0 | 0 | .200 [-.06, .46] |
| always_ask | .875 | .125 | 0 | 1.00 | .600 | .550 [.40, .69] |
| confidence_threshold | .850 | .150 | 0 | .49 | .131 | .602 [.41, .77] |
| entropy_threshold | .850 | .150 | 0 | .53 | .169 | .594 [.40, .76] |
| evpi | .825 | .031 | .144 | .69 | .350 | .657 [.51, .78] |
| noise_indep_voi | .806 | .025 | .169 | .65 | .312 | .659 [.51, .78] |
| **nm_voi** | .806 | .025 | .169 | .68 | .338 | .655 [.51, .78] |

Paired utility, NM-VoI minus other: never_ask +0.455 [.23, .70]; always_ask +0.105 [-.04, .23]; confidence_threshold +0.052 [-.04, .16];
entropy_threshold +0.061 [-.03, .17]; evpi -0.002 [-.06, .04]; **noise_indep_voi -0.004 [-.011, .000]**.

Val items only (10 items, 40 episodes; held out from the channel, belief and SNR fits): nm_voi .653, noise_indep_voi .660, evpi .740,
confidence_threshold .350, always_ask .250, never_ask .000. Thresholds were tuned on these same val items.

## 3. Held-out test (10 items, 40 episodes per policy) - PRELIMINARY
Single run after the protocol freeze (verified intact afterwards).

| policy | success | wrong | abstain | ask rate | unnecessary clar. | utility |
|---|---|---|---|---|---|---|
| never_ask | .500 | .500 | 0 | 0 | 0 | .000 [-.50, .55] |
| always_ask | .775 | .225 | 0 | 1.00 | .500 | .350 [-.15, .80] |
| confidence_threshold | .775 | .225 | 0 | .475 | .075 | .455 [-.03, .88] |
| entropy_threshold | .775 | .225 | 0 | .65 | .175 | .420 [-.09, .85] |
| evpi | .800 | .025 | .175 | .75 | .350 | .635 [.43, .81] |
| noise_indep_voi | .800 | .025 | .175 | .75 | .350 | .645 [.44, .82] |
| **nm_voi** | .800 | .025 | .175 | .725 | .350 | .653 [.45, .82] |

NM-VoI vs never_ask +0.653 [.25, 1.07]; vs always_ask +0.302 [.05, .63]; vs noise_indep_voi +0.007 [0.000, 0.022].

## 4. Ablations (dev utility; nm_voi .655)
open-only .491; closed-only .592; confirm-only .589; equal costs .613; uncalibrated belief (beta = 1) .596; no abstention .654.
Allowing all three question types beats open-only (paired +0.164 [.05, .30]); the other paired differences include zero. Removing abstention leaves utility unchanged but raises wrong answers (.106 vs .025).

## 5. What the results do and do not show
**Supported (preliminary):**
- VoI-style policies with abstention beat never-ask clearly and reduce wrong answers to about 3%. The trade is lower raw success (about .81 vs .85-.88 for always-ask and thresholds) and more clarification, including more "unnecessary" clarification (.31-.35 vs .13-.17 for the thresholds).
- Their utility advantage over always-ask and the thresholds is small and its dev intervals include zero; on the 10 val items alone the gap is larger but rests on 40 episodes.

**Not supported:**
- **NM-VoI did not outperform noise-independent VoI** on dev or test. The paired difference is about zero. The two differed in the first action in 5/160 dev clips (SNR 20: 3, 10: 1, 5: 1) and never in the final answer or abstention; utility differed only through interaction cost.
- Therefore these data do not show any benefit of modelling the noise dependence of the clarification channel.

**Why NM-VoI could not win here (partly verified, partly interpretation):**
- Verified: the channel is flat across SNR within sampling error, and the median expected-utility margin between the chosen action and the runner-up is 0.10.
- Interpretation: with a flat channel the SNR-conditioned and pooled channels give nearly identical expected utilities. Hand-picked human review (see [human_evaluation.md](human_evaluation.md)) points to noise-independent failure sources that would also mask an SNR effect.

## 6. Caveats specific to these numbers
- The "unnecessary clarification" rate is inflated by tie-breaking: 25 of 54 flagged NM-VoI dev episodes began from a near-tie, so it measures "asked when the arbitrary top choice was right", not user-perceived burden.
- Test has 10 items; intervals are wide and per-category cells hold 2-3 items.
- Robustness sweeps (cost scale, reply style, noise seed, second voice or noise type) were **not run**.
