# CLAUDE.md

Instructions for the coding agent working in this repository.

## Project goal

Research project: **Confidence-Calibrated Voice Agents That Know When to
Clarify** (Hind Alzahrani, Vizuara Voice Agents Bootcamp, 10 weeks).

The research question: when a voice agent is uncertain, the uncertainty can
come from acoustic misrecognition, intent ambiguity, or underspecification.
Are the available confidence signals calibrated well enough to tell these
apart, and can a calibrated signal drive a when-to-clarify policy that beats
both always-asking and never-asking on task success *and* user burden?

The deliverable is a paper, not a product. Code exists to produce defensible
numbers and honest figures.

## Layout

```
configs/    YAML run configs; configs/default.yaml is the reference
prompts/    the four prompts as plain text (answer, clarify, verbalized_conf, self_ask)
src/        library code (see README.md for the file-by-file map)
scripts/    build_data.py, run_pilot.sh, run_full.sh
experiments/log_template.md, plus one filled log per run
results/    generated; JSONL run logs and CSV metric tables (not committed)
data/       generated; audio, labels.jsonl, splits (not committed)
```

## Conventions

- **Signal convention**: every confidence signal is normalised to `[0, 1]`
  where **higher means more confident**. Semantic entropy and the
  self-interrogation flag are uncertainty measures, so they are inverted in
  `build_signals`. Preserve this invariant; policies and metrics depend on it.
- **Config over constants.** Anything you might vary between runs belongs in
  `configs/*.yaml`, not in a literal in the code.
- **Seeds.** `load_config` seeds Python, NumPy and torch. Do not add
  unseeded randomness.
- **Logging.** `run_agent.py` writes one JSON object per item with every
  signal, both branch outcomes, and latency. If you add a signal, add it to
  `ConfidenceSignals` so it lands in the log automatically.
- **Both branches, always.** `run_agent.py` runs the answer branch AND the
  clarify branch for every item, so that policy comparison stays pure
  post-processing. Do not "optimise" this away: it is what makes the
  `clarification_helps` label (the fusion target) computable.
- Type hints on new functions; docstrings that say *why*, not *what*.
- Keep modules small enough to read in one sitting.

## Hard rules

1. **Never fit anything on the test split.** Thresholds, temperature scaling
   and the fusion model are fit on `val` only. If you write code that touches
   test data before the final evaluation, that is a bug, not a shortcut.
2. **Never report accuracy without burden.** Any table or figure showing task
   success must show over-clarification rate beside it.
3. **Never silently change the evaluation set** once it is frozen (end of
   Week 4). If the data must change, say so explicitly and re-run everything.
4. **Do not invent numbers.** If an experiment has not been run, the cell
   stays `XX.X`. The paper draft marks placeholders deliberately.

## Current state

The scaffold is complete and import-clean. Metrics, policies, signal
extraction and the noise pipeline are implemented and unit-checked. What does
not exist yet is real data: `scripts/build_data.py` ships only the
hand-written underspecification set.

## Next steps, in order

1. **Data (Week 3-4).** Extend `scripts/build_data.py` with real spoken
   commands (SLURP or a MASSIVE slice) and with voiced AmbigQA / ClarQ-LLM
   items. Target 300 to 500 items, stratified across the four ambiguity types
   and the SNR levels. Freeze the split.
2. **Pilot.** `bash scripts/run_pilot.sh` on 20 items. Confirm every record
   has all five signals and that both branches produced sane text.
3. **Calibration (Week 5-6).** Run the full sweep; produce reliability
   diagrams per signal per noise level. The key question: does the best
   signal *change* between clean and noisy audio?
4. **Policies (Week 7).** Tune thresholds on val, fit the fusion, produce the
   success-versus-burden figure, pick an operating point.
5. **Robustness (Week 8).** SNR sweep, accent shift, adversarial homophones,
   one held-out domain. Report deltas honestly, including regressions.
6. **Write (Week 9-10).** The paper draft in the kit already has the section
   skeleton; results flow from the two CSVs.

## When asked to add a confidence signal

1. Add the field to `ConfidenceSignals` in `src/signals.py` (normalised, higher
   = more confident) and to `SIGNAL_NAMES`.
2. Populate it in `build_signals`.
3. Add a toggle under `signals:` in the config.
4. It then appears automatically in the log, the calibration table, and the
   fusion feature vector. Add a threshold policy for it in `build_policies`
   if you want it compared on its own.

## What not to do

- Do not add a heavyweight experiment framework. Plain scripts plus JSONL is
  deliberate and keeps the project reproducible by a third party.
- Do not commit audio, model weights, or `results/`.
- Do not replace the placeholder `XX.X` cells in the paper with guesses.
