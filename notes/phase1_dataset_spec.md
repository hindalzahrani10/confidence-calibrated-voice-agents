# Phase 1 feasibility dataset: specification (schema `phase1.v2`)

Status: **draft, unfrozen, unreviewed by any human.** 50 hand-authored synthetic base items, no audio.
This is a small feasibility dataset, not a statistically sufficient benchmark. Passing
`scripts/validate_phase1.py` shows internal consistency only; it does not show the annotations are correct.

Reproduce: `python3 scripts/author_phase1.py && python3 scripts/validate_phase1.py --src src`
(negative tests: `python3 scripts/test_validate_phase1.py`). `data/` is git-ignored, so the author script is
the committed source of truth. Pre-correction v1 files are kept locally in `data/phase1/archive/`.

## What changed from v1, and why

| v1 problem (audit) | v2 change |
|---|---|
| All 50 confirmations proposed the gold task and got "yes" | Each item has a correct probe (`yes`) and two incorrect probes (`no_bare`, `no_correction`) built from plausible listed alternatives. |
| A "no" had no recovery semantics | Open-world, strict-by-default recovery; rejections are stored as **exclusion evidence**, separate from recovered values. |
| Replies fixed per item, regardless of the question | Replies are reference scenarios for a documented simulator contract keyed on the actual question (below). |
| Gold option first in 86% of closed questions and always present | Gold position balanced (13/12/12) with a recorded seed; a gold-omitting question with a `none_of_these` reply exists for every ambiguous item. |
| `intent_09/10/11` were slot problems renamed as intents; `intent_03` was entity-sense ambiguity | Re-authored as genuine action ambiguity; every item now carries `distinguishing_axis` and the validator enforces it. |
| World-knowledge or contaminated replies (`acoustic_06/08/10/12`) | All replies are literal restatements of a recorded phrase (`say`); validator checks the phrase is in the reply text. |
| Implausible/incomplete alternatives (50 °C thermostat, `start_timer` with no duration, playlist-as-artist) | Replaced; every alternative is a complete task with valid slot values; >=2 alternatives per ambiguous item. |
| Only `jazz` accepted for "play some music" | `acceptable_tasks` with an explicit wildcard and a written justification. |
| `item_id` text named the category and SNR | Agent-visible ids are opaque (`u<hex>-<letter>`), letter order shuffled per item. |

## Agent-visible vs private

```
{ "schema_version", "base_id", "split", "source",
  "agent_visible": { "audio_variants": [ {variant_id, audio_path, audio_status} x4 ] },
  "private":       { everything else } }
```

* The future agent may use **only** `agent_visible` (audio, once it exists) and its own decisions.
* Private: `gold_task`, `alternatives`, `acceptable_tasks`, `probes` (including which proposal is correct and the
  expected replies), `recovery`, `utterance_slots`, `at_risk_slots`, `say`, the ambiguity label, the axis, the
  confusable, the SNR of each variant (`conditions`), `reference_text`. Top-level `base_id` is also treated as
  private because its text names the category: **Phase 2 must name audio files and shuffle record order by the
  opaque id**, never by `base_id`.
* The true SNR is private. The formulation conditions on SNR, so Phase 2 must decide whether the agent receives an
  *estimate* from the audio (assumed here) or the true value (an oracle). That is a research decision, not made here.
* Static check: `validate_phase1.py --src src` fails if agent-side code references private field names, and warns about
  legacy v1 names. It does not modify any runtime code.

## Probes are private reference scenarios

A probe is `(question, expected reply, recovery)`. It is **not** an instruction about what to ask or propose, and
the agent never sees it. Having both correct and incorrect probes does **not** make an evaluation unbiased: what
the agent actually proposes must depend only on its own observable evidence, and the simulator must judge that
actual proposal against the hidden gold task.

Per item: `open` (full-coverage; plus a partial-coverage question where >=2 slots are missing), `closed`
(one with the gold option, one without), `confirm` (correct/`yes`, incorrect/`no_bare`, incorrect/`no_correction`).
`none` items have no open/closed probes (`not_applicable_questions`).

## Recovery semantics (strict is primary)

Four things are kept apart:

1. **From the utterance** (`from_utterance`): slots the initial utterance supports. Excludes slots at acoustic risk,
   and includes `intent` only when intent is not itself uncertain.
2. **Explicitly supplied/affirmed by the reply** (`from_reply` = `supplies` + `confirms`).
3. **Rejected** (`exclusions`): a joint rejected proposal, or a rejected option set. Exclusions are never values.
4. **Unresolved** (`strict.unresolved_slots`).

Rules:

* Open-world: `candidate_set_exhaustive` is false everywhere, so rejecting alternatives never identifies the gold.
* `yes` to a correct proposal confirms every slot it asserts.
* `no_bare` and `none_of_these` supply nothing and can **never** be `full`.
* A rejected proposal that contradicts a slot the utterance supports means the agent demonstrably did not hold that
  slot, so it stops counting as known for that probe (this is why a bare "no" to "alarm at 6:30" is `none`, not `full`).
* `no_correction` resolves only the slots the user states.
* **Lenient** (sensitivity analysis only): slots of the rejected proposal that were not contradicted are treated as
  implicitly accepted. Stored in `recovery.lenient` with `assumed_accepted`; it differs from strict only for
  `no_correction`.
* Status: `full` = nothing unresolved; `partial` = the reply added information but something is unresolved;
  `none` = the reply added no information. Question types that do not apply are listed, not scored.

## Simulator contract (documented only; not implemented)

Input: hidden gold task (and `acceptable_tasks`), the item's `say` lexicon, `utterance_slots`/`at_risk_slots`, and the
agent's **actual** question as a structured object `{type, covers, proposed_task | options}` plus its text.
A seeded RNG breaks any tie.

1. **open**: reply only for `covers`, using the literal `say` phrase for each key; nothing about uncovered slots. If
   the question covers nothing answerable, return an `off_target` reply that conveys no information.
2. **closed**: if the gold value is among the options return exactly that option; otherwise `none_of_these`, never the
   nearest wrong option, and **no** value is revealed. Option order is the agent's own.
3. **confirm**: proposal equal to gold (on every asserted key) -> `yes`. Otherwise `no`, either bare or with a
   correction of the wrong keys (`correction_style` is a recorded experimental setting). Never `yes` to a wrong
   proposal; a proposal asserting only some slots is judged on the slots it asserts.
4. Malformed or compound questions are answered per covered key; the simulator never volunteers uncovered slots.
5. Scoring uses a structured comparison of `(intent, slots)` against `acceptable_tasks`, never substring matching.

## Prevalence and reporting

37 of 50 items are labelled as needing clarification (13 clear). One of them (`underspec_03`) has a wildcard
`acceptable_tasks`, so asking is not required for success there. Future reporting should include: results by
ambiguity category; overall task success; unnecessary-clarification rate; clarification burden (over-clarification
rate beside any success figure); and a clearly labelled prevalence-reweighted analysis. With 2-3 items per
category-by-split cell, per-category numbers are anecdotal.

## Assumptions and limitations that could affect publication

* Every annotation is a single AI-drafted pass (`annotator_ids: ["claude_draft"]`, `agreement: null`). No human
  review, agreement score or reliability exists. `author` and `validator` were written by the same model and share
  blind spots (the recovery rule is deliberately implemented twice, but by the same author).
* The intent-vs-slot boundary is a judgement (operational rule: alternatives call a different action, not a different
  argument of one action). `intent_01`, `intent_04`, `intent_07` are boundary cases.
* Gold intents of intent items are arbitrary picks among defensible readings; `gold_prior` (modal/minority/unclear)
  is the author's unmeasured guess.
* `acoustic` is a design hypothesis, unverified until a TTS+ASR check in Phase 2; `acoustic_10` is a weak pair. At
  clean SNR an acoustic item may involve no acoustic uncertainty at all, so analyses should use measured ASR
  confidence, not the label.
* Questions are template-rendered: literal and consistent but stilted; naturalness is unrated by humans.
* Replies are idealised literal restatements. Real users paraphrase, hesitate, and mis-hear the agent's question;
  those conditions are not represented. The "world-knowledge" reply condition is deliberately deferred.
* Closed questions always have 3 options (gold-containing) or 2 (gold-omitting); option-count variation is untested.
* The legacy pipeline cannot consume this file as-is: `src/run_agent.py` pairs one fixed reply with any generated
  question and scores by substring match on `gold_answers`. A Phase 2 adapter is required; nothing in `src/` was
  changed.
* The set is **not frozen** (`status: draft_unfrozen`). Under hard rule 3, freezing is a separate, explicit step.
