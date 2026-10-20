# Experiment log

## Run ID: `2026-10-08-qwen-pilot-debug`

**Date:** 2026-10-08
**Config:** `configs/pilot_debug.yaml` (Local M1 Mac)
**Commit:** `[Insert your latest commit hash]`
**Seed(s):** 42

### Question
How do existing baseline uncertainty policies and text-based EVPI handle underspecified spoken requests across varying SNR levels?

### Prediction (write before running)
Baseline text-based policies will misidentify acoustic ASR errors as semantic ambiguity, leading to unhelpful clarification loops that fail to resolve the user's request. 

### Setup
| Field | Value |
| --- | --- |
| ASR model | Whisper-base |
| LLM | Qwen2.5-1.5B-Instruct |
| Eval items (val / test) | 8 / 12 |
| Ambiguity types present | underspecified |
| SNR levels | Clean, 20 dB, 10 dB, 5 dB |
| Samples per item (k) | 1 |
| Runtime | ~380 seconds |

### Policy results (-> Table 1)
| Policy | Success ↑ | Ask Rate (Burden) ↓ | Clarif. prec. ↑ | Clarif. recall ↑ | Mean Turns ↓ |
| --- | --- | --- | --- | --- | --- |
| Never clarify | 0.00 | 0.00 | N/A | 0.00 | 1.00 |
| Always clarify | 0.67 | 1.00 | 1.00 | 1.00 | 2.00 |
| ASR-confidence threshold | 0.00 | 0.25 | 1.00 | 0.25 | 1.25 |
| Token log-prob threshold | 0.00 | 0.08 | 1.00 | 0.08 | 1.08 |
| Verbalized threshold | 0.33 | 0.50 | 1.00 | 0.50 | 1.50 |
| Semantic-entropy threshold | 0.50 | 0.83 | 1.00 | 0.83 | 1.83 |
| Self-interrogation | 0.33 | 0.58 | 1.00 | 0.58 | 1.58 |
| Calibrated fusion (ours) | 0.33 | 0.42 | 1.00 | 0.42 | 1.42 |

### Signal calibration (-> Figure 2b)
| Signal | Clean ECE | SNR 20 dB | SNR 10 dB | SNR 5 dB | AUROC (all) |
| --- | --- | --- | --- | --- | --- |
| ASR confidence | 0.866 | 0.836 | 0.835 | 0.766 | NaN |
| Token log-prob | 0.927 | 0.916 | 0.891 | 0.917 | NaN |
| Verbalized | 0.710 | 0.780 | 0.780 | 0.710 | NaN |
| Semantic entropy | 0.480 | 0.433 | 0.302 | 0.280 | NaN |
| Self-interrogation | 0.400 | 0.400 | 0.600 | 0.600 | NaN |

### Per-ambiguity-type breakdown
| Ambiguity type | n | Success (best policy) | Burden | Notes |
| --- | --- | --- | --- | --- |
| underspecified | 20 | 0.67 | 1.00 | All items were underspecified variants. The "Always clarify" policy performed best but resulted in maximum burden. |

### What actually happened
The prediction held. As noise increased, the agent generated clarification questions that were either nonsensical or failed to address the actual missing information. Furthermore, even in clean conditions, the Qwen2.5-1.5B model struggled to generate useful clarifying questions for simple intents, often answering incorrectly despite receiving the user's clarification.

### Failure taxonomy (sample ~10 transcripts by hand)
| Category | Count | Example item_id |
| --- | --- | --- |
| Confidently wrong on misheard input | 3 | `underspec_003__snr10` |
| Clarifying question itself unclear/illogical | 4 | `underspec_003` |
| Ambiguity missed (answered when unclear) | 6 | `underspec_000` |

**Sample Transcript Failure Cases:**
1. **Item `underspec_000` (Clean Audio)**
   * **Transcript:** "Set an alarm."
   * **Agent Asked:** "Do you want to set an alarm for today or tomorrow?"
   * **User Replied:** "seven in the morning"
   * **Result:** Agent set it for 7 AM but never resolved whether it was for today or tomorrow. Task failed. 

2. **Item `underspec_003` (Clean Audio)**
   * **Transcript:** "Book a table."
   * **Agent Asked:** "Do you need a reservation or do you want to book an actual table?" (Illogical question generation).
   * **User Replied:** "two people at eight"
   * **Result:** Agent confirmed a table for 8 PM but failed to ask for a restaurant or location. Task failed.

3. **Item `underspec_003__snr10` (10 dB Noise)**
   * **Transcript:** "Bucket Table." (Severe ASR Error).
   * **Agent Asked:** "Could you please specify if you mean 'Bucket' or 'Table'?"
   * **User Replied:** "two people at eight"
   * **Result:** Agent hallucinated that "The user is asking for information about a table with two chairs arranged around it." Complete contextual collapse due to noise.

### Threats to this result
Small sample size (n=20) and use of a smaller model (1.5B) limit the generalizability of the exact success rates. The LLM's weak reasoning capability may be conflating standard ambiguity resolution with instruction-following failures.
