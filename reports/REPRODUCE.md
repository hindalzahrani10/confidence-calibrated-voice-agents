# Reproducing the NM-VoI feasibility experiments

All data are SYNTHETIC (hand-authored text, TTS speech, a real noise recording used as background). Human semantic review of the
Phase 1 annotations is deferred; nothing here is human-validated. Everything runs on the existing 2-vCPU / 3.8 GB host, no GPU,
no paid service. Paths are relative to the repository root.

## 0. Environment
```bash
python3 -m venv .venv && . .venv/bin/activate
pip install --no-cache-dir numpy scipy soundfile tqdm pyyaml matplotlib piper-tts faster-whisper
# models/assets (free; sizes and sha256 recorded in the manifests)
#   Piper voice  en_US-lessac-medium  -> data/models/piper/      (63 MB)     rhasspy/piper-voices on Hugging Face
#   faster-whisper-small              -> data/models/faster-whisper-small/   (484 MB)   Systran/faster-whisper-small
#   DEMAND PCAFETER_16k.zip (CC BY 4.0, Zenodo 1227121) -> member PCAFETER/ch01.wav -> data/noise/DEMAND_PCAFETER_ch01_16k.wav
```
Versions used are recorded in `data/phase2/audio.v1/manifest.json` and `data/phase3/asr.v1/config.json`.

## 1. Phase 1 (dataset) -- verify, snapshot
```bash
python3 scripts/author_phase1.py            # deterministic; regenerates data/phase1/*
python3 scripts/validate_phase1.py --src src
TMPDIR=<scratch> python3 scripts/test_validate_phase1.py     # 38 negative tests
python3 scripts/snapshot_phase1.py          # read-only snapshot phase1.v2.0+<hash>
```

## 2. Phase 2 (audio)
```bash
python scripts/generate_phase2_audio.py && python scripts/generate_phase2_audio.py --verify 40
python scripts/validate_phase2_audio.py && TMPDIR=<scratch> python scripts/test_validate_phase2_audio.py   # 18 negative tests
```

## 3. Phase 3 (ASR, task belief, calibration)
```bash
python scripts/run_asr.py                       # resumable; ~6 s/clip on this host (~1.9 h for 1132 clips)
python scripts/build_catalogue.py
python scripts/phase3_task_inference.py         # fits the belief temperature on TRAIN; reports train/val only
python scripts/fit_snr_estimator.py             # blind SNR bucket rule, centroids from TRAIN clips
```

## 4. Phase 4-5 (channel, simulator, policies)
```bash
python scripts/estimate_channel.py              # channel table from TRAIN replies
python scripts/test_nmvoi.py                    # analytic, monotonicity and leakage tests
python scripts/run_episodes.py --splits train val --out data/phase5/dev.v1     # may log on-demand reply requests
python scripts/generate_ondemand.py             # only after run_asr.py has finished
python scripts/tune_policies.py                 # tau / entropy thresholds on VAL only
python scripts/run_episodes.py --splits train val --out data/phase5/dev.v1
```

## 5. Freeze, then the single held-out evaluation
```bash
python scripts/freeze_protocol.py               # writes reports/protocol_freeze.json (hashes of code, config, tuned values)
python scripts/run_episodes.py --splits test --out data/phase5/test.v1
python scripts/generate_ondemand.py && python scripts/run_episodes.py --splits test --out data/phase5/test.v1
python scripts/freeze_protocol.py --verify
python scripts/analyze_results.py --episodes data/phase5/test.v1/episodes.jsonl --tag test
```
Scripts refuse the `test` split unless `reports/protocol_freeze.json` exists.

## Determinism
Every clip, noise slice and tie-break derives from `sha256(master seed | purpose | id)`; no global RNG is used.
`generate_phase2_audio.py --verify N` regenerates N clips and compares bytes. ASR uses temperature 0, int8 CPU.
