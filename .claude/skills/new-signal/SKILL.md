---
name: new-signal
description: Add a candidate signal to the pipeline, with its economic prior written before any data are seen.
argument-hint: "<feature or idea, e.g. 'mom40' or 'Amihud illiquidity'>"
disable-model-invocation: true
---

Add the signal: $ARGUMENTS

Do the steps in order. Do not compute IC or look at returns at any step.

1. Prior first. State the family, the sign (+1 or -1), a one-sentence economic rationale and a reference. Check `prior_for('{feature}_{method}')` in src/signals/priors.py. The name rules match by prefix (`mom`, `vol`, `beta`, `log_adv`), so confirm that the rule that matches is the one you mean. If no rule or a wrong rule matches, add or fix the rule. Ask the user to confirm the prior before you continue. If you cannot give a rationale, stop.
2. Check for a duplicate: SIGNAL_SPECS in scripts/pipeline/04_generate_signals.py, including the commented lines that were tried and switched off.
3. Feature. If features.parquet already has the column, skip this step. For a new window, change WINDOWS in scripts/pipeline/03_build_features.py. For a new formula, add a function in src/features/, call it from feature_pipeline.build_features, and add it to the no-lookahead test in tests/test_features.py. A feature at date t uses data up to the close of t.
4. Add one line to SIGNAL_SPECS: `(feature, method, params)`. Methods: zscore_tanh, zscore_clip, zscore, rank, decile. Never delete a line to hide a trial; comment it out.
5. Run `/Users/lemoal/miniforge3/envs/quantlab311/bin/python -m pytest -q`.
6. Use the leakage-auditor agent on the change.
7. Report the change, the number of trials so far (active plus commented lines in SIGNAL_SPECS, plus one for each earlier removed variant you know of), and the next command: `/run-eval <new_tag>`. If step 3 changed features, say that 03_build_features.py must run first and that it overwrites features.parquet, so ask before you run it.
