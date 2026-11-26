---
name: run-eval
description: Generate signals (stage 04) and evaluate them (stage 05) under a new run_id, then review the result. Never touches the holdout.
argument-hint: "<tag> [YYYY-MM-DD]"
disable-model-invocation: true
---

Arguments: $ARGUMENTS (first: tag; second, optional: date, default today).

1. Check that `data/clean/{asset}/{region}/{universe}/{freq}/runs/{date}_{tag}` does not exist (the dataset is in config/dataset_config.json, or in QAL_DATASET_CONFIG). If one exists, choose a new tag and say so. Never overwrite or delete an old run.
2. Run, with PY=/Users/lemoal/miniforge3/envs/quantlab311/bin/python:
   - `$PY scripts/pipeline/04_generate_signals.py --date <date> --tag <tag>`
   - `$PY scripts/pipeline/05_evaluate_signals.py --date <date> --tag <tag> --n_trials_extra <N>`
   N = the number of variants tried and removed: the commented specs in SIGNAL_SPECS of 04, plus others you know of. State N and how you counted it.
3. Keep the default strict mode and thresholds. Change one only if the user asks, and then label the result as exploration.
4. Do not run stages 06 to 09, and do not read holdout results.
5. Give the run_id to the signal-reviewer agent. Paste into its prompt the gate table and the IC-by-year table that 05 printed.
6. Report: run_id, admitted signals, the gate that stopped the others, and the reviewer's verdict. No admitted signal is a valid result.
