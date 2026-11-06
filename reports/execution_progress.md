# NM-VoI execution progress

Updated: see the last entry. All data is SYNTHETIC (hand-authored text, TTS speech). Human semantic review is deferred;
nothing here is human-validated. No paid resource has been used or provisioned. Nothing is committed.

## Environment (existing EC2 instance only)
2 vCPU, 3.8 GB RAM, no GPU, ~0.7 GB free disk after installs. Python venv at `.venv/` (git-ignored).
Installed from PyPI (free): numpy, scipy, soundfile, matplotlib, tqdm, pyyaml, piper-tts, faster-whisper (+ deps).
Downloaded (free, hash-recorded): Piper `en_US-lessac-medium` (63 MB), faster-whisper-small (484 MB),
DEMAND PCAFETER ch01 (CC BY 4.0, 9.6 MB kept; 107 MB zip deleted).
No torch / no LLM: a 7B model cannot fit this host, so the agent's task belief is a transparent lexical model (see Phase 3).

## Phase 1: verified, not recreated
Validator OK; 38/38 negative tests; snapshot `data/phase1/snapshots/phase1.v2.0+05471eebec39` (read-only, hashed).
Decisions still open from Phase 1: the contradiction rule (applied as documented) -- see decisions below.

## Phase 2: audio generation  (DONE)
`data/phase2/audio.v1`: 1132 clips = 200 initial (50 x {clean,20,10,5}) + 932 reply (233 x 4); 283 clean TTS syntheses.
Achieved SNR within +/-0.01 dB of nominal; no clipping. Validator OK; 18/18 negative tests; determinism re-check 40/40.

## Phase 3: ASR  (RUNNING in background)
`scripts/run_asr.py` -> `data/phase3/asr.v1/asr_results.jsonl` (append-only, resumable; ~6 s/clip, ~1.9 h total).
Log: `data/phase3/asr_run.log`.  Re-launch is safe (skips finished clips). Do NOT start a second instance.

## Phase 3-5 code written (tests passing: scripts/test_nmvoi.py ALL PASSED)
`src/ontology.py` (agent: patterns, catalogue loader, lexical belief), `src/channel.py` (reply parsers, blind SNR, channel table),
`src/nmvoi.py` (EU machinery + 7 baselines/ablations), `src/simulator.py` + `src/audio_service.py` (evaluator side),
`src/evalutil.py` (metrics), scripts: build_catalogue, phase3_task_inference, estimate_channel, run_episodes, generate_ondemand.
Agent/evaluator separation enforced by an AST identifier scan in scripts/test_nmvoi.py.
Grammar frozen (hash recorded in data/phase3/calibration.v1.json) BEFORE looking at val/test. Test split sealed:
scripts refuse `test` unless reports/protocol_freeze.json exists.

## Decisions made autonomously (defaults I recommended earlier; none needed spending)
- Noise: one real recording (DEMAND cafeteria), split into disjoint train/val/test time ranges. White noise not run.
- One TTS voice. Reply noise replicates = 1 (ASR cost). SNR defined on active-speech power. Digital-silence padding.
- Agent SNR knowledge: estimated blindly from the audio (lead-in noise floor), never the true label.

## Decisions requiring approval / blockers
- (none that block independent work yet)

## Update: Phase 3 ASR finished; channel estimated (TRAIN only)
ASR complete: 1132/1132 clips. `data/phase4/channel.v1.json` (train replies). Smoothed r by bucket clean/20/10/5:
confirm_yes 1.00 all; confirm_no .92/.97/.97/.95; closed .95/.95/.93/.89; open .90/.94/.82/.79.
Noise effects on replies are small except open answers. First dev episode pass logged 137 on-demand reply requests
(partial results printed there are NOT final: pending episodes were excluded). generate_ondemand.py launched once (background).

## Update: dev stage complete (train+val, 40 items x 4 SNR, 2080 episodes, 0 pending)
Tuned on val only: tau=0.4, h=0.75 (data/phase5/tuned.json). Dev (pooled SNR) utility: nm_voi .655, noise_indep_voi .659, evpi .657,
confidence_threshold .602, always_ask .550, never_ask .200. NM-VoI is NOT distinguishable from noise-independent VoI on dev
(channel is nearly noise-flat). VoI family trades success (.81) for far fewer wrong answers (.03) via abstention.
Results: reports/results/dev/. Next: freeze protocol, single test run (preliminary, 10 items).

## Update: test run done (single, frozen), report written
Freeze verified intact. Test (10 items, PRELIMINARY): nm_voi util .653, noise_indep .645, evpi .635, conf_thr .455, always .350, never 0.
Report: reports/nmvoi_experimental_report.md. Robustness sweeps not run (documented there).
