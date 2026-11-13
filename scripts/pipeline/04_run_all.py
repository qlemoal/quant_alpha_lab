#!/usr/bin/env python
'''
===============================================
Run the Full Pipeline (or some phases of it)
===============================================

WHAT IT DOES
    Runs the phase scripts one after the other. Each phase is a normal script that you can also run by hand.
    Before a phase starts, the runner checks that the files this phase reads exist.
    If a phase fails, the runner stops. Later phases do not run on stale inputs.

THE PIPELINE

    04a  generate signals        features.parquet, ohlcv.parquet  ->  signals/
     |
    04b  evaluate signals        signals/                         ->  signal_eval/   (IC, FDR, selection)
     |
    04c  fit the combiner        signals/, signal_eval/           ->  en_scores/, en_model/
     |
    04d  estimate risk model     en_scores/, features.parquet     ->  risk_model/
     |
    04e  construct weights       en_scores/, risk_model/          ->  weights/
     |
    04f  run the backtest        weights/, ohlcv, features        ->  backtest/      (judge the HOLDOUT block)

    Every stage folder is {clean_base}/{stage}/{run_id}/, with run_id = {date}_{tag}.

USAGE
    python scripts/pipeline/04_run_all.py --date 2026-11-07 --tag default                 # all phases
    python scripts/pipeline/04_run_all.py --date 2026-11-07 --tag default --from 04d      # from 04d to the end
    python scripts/pipeline/04_run_all.py --date 2026-11-07 --tag default --only 04f      # one phase
    python scripts/pipeline/04_run_all.py --date 2026-11-07 --tag default --check         # inputs only, run nothing

NOTES
    - --from re-uses the outputs of the earlier phases as they are on disk. If you changed an earlier phase
      (code or config), start from that phase, not later.
    - The phases of one run_id overwrite their own outputs. Use a new --tag to keep a previous result.
'''

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))          # `import config...` works wherever the script is started from

from config.paths import get_paths, run_id_from_args


# =============================================================================
# THE PHASES
# =============================================================================

PHASES = [
    ('04a', '04a_generate_signals.py',     'Generate signals'),
    ('04b', '04b_evaluate_signals.py',     'Evaluate signals (IC, Newey-West, FDR, selection)'),
    ('04c', '04c_fit_combiner.py',         'Fit the Elastic Net combiner (CPCV + holdout)'),
    ('04d', '04d_estimate_risk_model.py',  'Estimate the risk model (covariance, clusters)'),
    ('04e', '04e_construct_weights.py',    'Construct the portfolio weights'),
    ('04f', '04f_run_backtest.py',         'Backtest with trading costs'),
]
PHASE_IDS = [p[0] for p in PHASES]


def required_inputs(phase_id: str, cfg) -> list:
    '''
    Files that a phase reads. They must exist before the phase starts.
    Keep this list equal to what the phase script really opens.
    '''
    signals = cfg.stage_dir('signals')
    en_scores = cfg.stage_dir('en_scores') / 'en_scores.parquet'
    risk = cfg.stage_dir('risk_model')
    weights = cfg.stage_dir('weights')
    return {
        '04a': [cfg.features_file, cfg.ohlcv_file],
        '04b': [signals / 'signals.parquet', signals / 'config.json'],
        '04c': [signals / 'signals.parquet', cfg.stage_dir('signal_eval') / 'selected_signals.json'],
        '04d': [en_scores, cfg.features_file],
        '04e': [en_scores, risk / 'cov.parquet', risk / 'clusters.parquet', risk / 'summary.json'],
        '04f': [weights / 'weights.parquet', weights / 'weights_diag.parquet', cfg.ohlcv_file, cfg.features_file],
    }[phase_id]


def missing_inputs(phase_id: str, cfg) -> list:
    return [str(f) for f in required_inputs(phase_id, cfg) if not Path(f).exists()]


# =============================================================================
# RUNNING
# =============================================================================

def run_phase(phase_id: str, script: str, title: str, date: str, tag: str) -> bool:
    '''Check the inputs, run the script as a subprocess, return True if it succeeded.'''
    cfg = get_paths(run_id=run_id_from_args(date, tag))
    print(f'\n{"=" * 70}\nPHASE {phase_id}: {title}\n{"=" * 70}')

    missing = missing_inputs(phase_id, cfg)
    if missing:
        print('FAILED: inputs missing. Run the earlier phases first:')
        for m in missing:
            print(f'  - {m}')
        return False

    cmd = [sys.executable, str(ROOT / 'scripts' / 'pipeline' / script), '--date', date, '--tag', tag]
    print(f'Command: {" ".join(cmd)}\n')
    env = {**os.environ, 'PYTHONPATH': str(ROOT) + os.pathsep + os.environ.get('PYTHONPATH', '')}
    start = time.time()
    result = subprocess.run(cmd, cwd=ROOT, env=env)
    minutes = (time.time() - start) / 60

    if result.returncode != 0:
        print(f'\nFAILED: phase {phase_id} exited with code {result.returncode} after {minutes:.1f} min')
        return False
    print(f'\nOK: phase {phase_id} finished in {minutes:.1f} min')
    return True


def print_headline(cfg) -> None:
    '''Print the holdout line of the backtest, so that the first number you see is the honest one.'''
    metrics_file = cfg.stage_dir('backtest') / 'metrics.json'
    if not metrics_file.exists():
        return
    with open(metrics_file) as f:
        metrics = json.load(f)
    block = metrics.get('holdout') or metrics.get('full')
    name = 'HOLDOUT' if metrics.get('holdout') else 'FULL PERIOD (no holdout reported)'
    if block and block.get('sharpe') is not None:
        lo, hi = block['sharpe_ci95']
        print(f'\n{name}: net Sharpe {block["sharpe"]:.2f}  (95% interval {lo:.2f} to {hi:.2f}),'
              f' annual return {block["ann_return"]:.1%}, max drawdown {block["max_drawdown"]:.1%}')


def main():
    parser = argparse.ArgumentParser(description='Run the pipeline phases 04a to 04f')
    parser.add_argument('--date', type=str, required=True, help='Run date, YYYY-MM-DD (used in the run_id)')
    parser.add_argument('--tag', type=str, default='default', help='Run tag (default: default)')
    parser.add_argument('--from', dest='from_phase', type=str, default='04a', choices=PHASE_IDS,
                        help='First phase to run (default: 04a)')
    parser.add_argument('--only', dest='only_phase', type=str, choices=PHASE_IDS, help='Run this phase only')
    parser.add_argument('--check', action='store_true', help='Check the inputs of the phases and run nothing')
    args = parser.parse_args()

    if not re.fullmatch(r'\d{4}-\d{2}-\d{2}', args.date):
        parser.error(f'--date must be YYYY-MM-DD, got {args.date!r}')
    if args.only_phase and args.from_phase != '04a':
        parser.error('use --only or --from, not both')

    run_id = run_id_from_args(args.date, args.tag)
    cfg = get_paths(run_id=run_id)
    phases = [args.only_phase] if args.only_phase else PHASE_IDS[PHASE_IDS.index(args.from_phase):]

    print('=' * 70)
    print('PIPELINE RUNNER')
    print('=' * 70)
    print(f'Run ID: {run_id}')
    print(f'Phases: {", ".join(phases)}')

    # --check: report the inputs of every phase. A phase whose inputs are made by an earlier phase of
    # this same call is reported as "missing", because nothing has run yet.
    if args.check:
        print('\nInput check:')
        for pid in phases:
            missing = missing_inputs(pid, cfg)
            print(f'  {pid}: {"ok" if not missing else "MISSING"}')
            for m in missing:
                print(f'      - {m}')
        return

    start = time.time()
    for pid in phases:
        _, script, title = next(p for p in PHASES if p[0] == pid)
        if not run_phase(pid, script, title, args.date, args.tag):
            print(f'\nPIPELINE STOPPED at phase {pid}.')
            sys.exit(1)

    print('\n' + '=' * 70)
    print(f'ALL PHASES DONE in {(time.time() - start) / 60:.1f} min')
    print(f'Results: {cfg.stage_dir("backtest")}/')
    print_headline(cfg)
    print('=' * 70)


if __name__ == '__main__':
    main()