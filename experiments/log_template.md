# Experiment log

One entry per run. Copy the template below, fill it in *before* you look at
the results, and keep the entries in chronological order. The rows of the
policy table map one-to-one onto Table 1 of the paper, and the signal table
maps onto the ablation grid (Figure 2b).

Writing the prediction down before seeing the numbers is what turns a sweep
into an experiment. It is also what makes a surprising result recognisable as
a finding instead of a bug.

---

## Run ID: `YYYY-MM-DD-short-name`

**Date:** 
**Config:** `configs/______.yaml`  
**Commit:** `______`  
**Seed(s):** 

### Question
One sentence. What does this run decide?

### Prediction (write before running)
What you expect, and roughly by how much. If you are wrong, that is the
interesting case.

### Setup
| Field | Value |
| --- | --- |
| ASR model | Whisper-____ |
| LLM | ____ |
| Eval items (val / test) | ____ / ____ |
| Ambiguity types present | none / acoustic / intent / underspecified |
| SNR levels | ____ |
| Samples per item (k) | ____ |
| Runtime | ____ |

### Policy results (-> Table 1)

Copy from `results/policy_comparison.csv`. Success and burden in percent,
with bootstrap 95% CIs.

| Policy | Success ↑ | Burden ↓ | ECE ↓ | Clarif. prec. ↑ | Clarif. recall ↑ | Turns ↓ |
| --- | --- | --- | --- | --- | --- | --- |
| Never clarify | | | | | | |
| Always clarify | | | | | | |
| ASR-confidence threshold | | | | | | |
| Token log-prob threshold | | | | | | |
| Verbalized threshold | | | | | | |
| Semantic-entropy threshold | | | | | | |
| Self-interrogation | | | | | | |
| Calibrated fusion (ours) | | | | | | |

### Signal calibration (-> Figure 2b)

Copy from `results/signal_calibration.csv`.

| Signal | Clean ECE | SNR 20 dB | SNR 10 dB | Accented | AUROC (all) |
| --- | --- | --- | --- | --- | --- |
| ASR confidence | | | | | |
| Token log-prob | | | | | |
| Verbalized | | | | | |
| Semantic entropy | | | | | |
| Fused (ours) | | | | | |

### Per-ambiguity-type breakdown

| Ambiguity type | n | Success (best policy) | Burden | Notes |
| --- | --- | --- | --- | --- |
| none | | | | |
| acoustic | | | | |
| intent | | | | |
| underspecified | | | | |

### What actually happened
Three sentences. Did the prediction hold?

### Failure taxonomy (sample ~10 transcripts by hand)
| Category | Count | Example item_id |
| --- | --- | --- |
| Confidently wrong on misheard input | | |
| Unnecessary hedging (asked when clear) | | |
| Ambiguity missed (answered when unclear) | | |
| Clarifying question itself unclear | | |
| Other | | |

### Threats to this result
Anything that could make the number wrong: too few items, a scorer artefact,
a threshold that landed on a cliff, a split that leaked.

### Decision
What changes next because of this run.

---
