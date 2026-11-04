#!/usr/bin/env python
'''
===============================================
Run Full Pipeline (or individual phases)
===============================================

Sequentially runs phases 0a -> 0b -> 0c -> 1 -> 2 -> 3.
Each phase is a separate script you can also run manually.

Usage:
  python scripts/pipeline/04_run_all.py --date 2026-10-29 --tag default  # Run all
  python scripts/pipeline/04_run_all.py --date 2026-10-29 --tag default --from 1  # Run from phase 1
  python scripts/pipeline/04_run_all.py --date 2026-10-29 --tag default --only 2  # Run only phase 2
  python scripts/pipeline/04_run_all.py --date 2026-10-29 --tag default --check   # Check prerequisites only
'''

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import date

PHASES = [
    ('0a', '04a_generate_signals.py', 'Generate signals (make_signal)'),
    ('0b', '04b_evaluate_signals.py', 'Evaluate signals (IC, FDR, decay)'),
    ('0c', '04c_fit_combiner.py', 'Fit Elastic Net combiner'),
    ('1',  '04d_estimate_risk_model.py', 'Estimate risk model (cov + clusters)'),
    ('2',  '04e_construct_weights.py', 'Construct portfolio weights'),
    ('3',  '04f_run_backtest.py', 'Run backtest with costs'),
]

# New structure: clean/equities/US/SP500/daily/{step}/{run_id}/{data_file}.parquet + config.json
# run_id format: YYYY-MM-DD_tag (e.g., 2026-10-29_default)

def find_run_dir(step: str, run_id: str) -> "Path | None":
    """Find the run directory for a given step and run_id."""
    base = Path(f'data/clean/equities/US/SP500/daily/{step}')
    if not base.exists():
        return None
    # Look for exact match first
    exact = base / run_id
    if exact.exists():
        return exact
    # Fallback: find by tag suffix
    tag = run_id.split('_', 1)[1] if '_' in run_id else run_id
    matches = list(base.glob(f'*_{tag}'))
    if matches:
        return sorted(matches)[-1]  # latest
    return None


def check_prereqs(phase_id: str, run_id: str):
    """Check if prerequisites for a phase exist."""
    missing = []
    tag = run_id.split('_', 1)[1] if '_' in run_id else run_id
    
    # Fixed paths for single-file base tables
    ohlcv_file = Path('data/clean/equities/US/SP500/daily/ohlcv/ohlcv.parquet')
    features_file = Path('data/clean/equities/US/SP500/daily/features/features.parquet')
    
    if phase_id == '0a':
        # Need features and OHLCV
        if not features_file.exists():
            missing.append('data/clean/equities/US/SP500/daily/features/features.parquet')
        if not ohlcv_file.exists():
            missing.append('data/clean/equities/US/SP500/daily/ohlcv/ohlcv.parquet')
    
    elif phase_id == '0b':
        # Need signals directory
        signals_dir = find_run_dir('signals', run_id)
        if not signals_dir or not (signals_dir / 'signals.parquet').exists():
            missing.append(f'data/clean/equities/US/SP500/daily/signals/{run_id}/signals.parquet')
    
    elif phase_id == '0c':
        # Need signals and signal_eval
        signals_dir = find_run_dir('signals', run_id)
        signal_eval_dir = find_run_dir('signal_eval', run_id)
        if not signals_dir or not (signals_dir / 'signals.parquet').exists():
            missing.append(f'data/clean/equities/US/SP500/daily/signals/{run_id}/signals.parquet')
        if not signal_eval_dir or not (signal_eval_dir / 'selected_signals.json').exists():
            missing.append(f'data/clean/equities/US/SP500/daily/signal_eval/{run_id}/selected_signals.json')
    
    elif phase_id == '1':
        # Need en_scores, ohlcv, features
        en_scores_dir = find_run_dir('en_scores', run_id)
        if not en_scores_dir or not (en_scores_dir / 'en_scores.parquet').exists():
            missing.append(f'data/clean/equities/US/SP500/daily/en_scores/{run_id}/en_scores.parquet')
        if not ohlcv_file.exists():
            missing.append('data/clean/equities/US/SP500/daily/ohlcv/ohlcv.parquet')
        if not features_file.exists():
            missing.append('data/clean/equities/US/SP500/daily/features/features.parquet')
    
    elif phase_id == '2':
        # Need risk_model, en_scores
        risk_dir = find_run_dir('risk_model', run_id)
        en_scores_dir = find_run_dir('en_scores', run_id)
        if not risk_dir or not (risk_dir / 'cov.parquet').exists():
            missing.append(f'data/clean/equities/US/SP500/daily/risk_model/{run_id}/cov.parquet')
        if not risk_dir or not (risk_dir / 'clusters.parquet').exists():
            missing.append(f'data/clean/equities/US/SP500/daily/risk_model/{run_id}/clusters.parquet')
        if not risk_dir or not (risk_dir / 'summary.json').exists():
            missing.append(f'data/clean/equities/US/SP500/daily/risk_model/{run_id}/summary.json')
        if not en_scores_dir or not (en_scores_dir / 'en_scores.parquet').exists():
            missing.append(f'data/clean/equities/US/SP500/daily/en_scores/{run_id}/en_scores.parquet')
    
    elif phase_id == '3':
        # Need weights, en_scores, ohlcv, features
        weights_dir = find_run_dir('weights', run_id)
        en_scores_dir = find_run_dir('en_scores', run_id)
        if not weights_dir or not (weights_dir / 'weights.parquet').exists():
            missing.append(f'data/clean/equities/US/SP500/daily/weights/{run_id}/weights.parquet')
        if not en_scores_dir or not (en_scores_dir / 'en_scores.parquet').exists():
            missing.append(f'data/clean/equities/US/SP500/daily/en_scores/{run_id}/en_scores.parquet')
        if not ohlcv_file.exists():
            missing.append('data/clean/equities/US/SP500/daily/ohlcv/ohlcv.parquet')
        if not features_file.exists():
            missing.append('data/clean/equities/US/SP500/daily/features/features.parquet')
    
    return missing


def run_phase(phase_id: str, script: str, date: str, tag: str):
    run_id = f'{date}_{tag}'
    print(f'\n{"="*60}')
    print(f'RUNNING PHASE {phase_id}: {script}')
    print(f'{"="*60}')

    missing = check_prereqs(phase_id, run_id)
    if missing:
        print(f'✗ Prerequisites not met:')
        for m in missing:
            print(f'  - {m}')
        return False

    # All pipeline scripts now use --date and --tag
    cmd = [sys.executable, f'scripts/pipeline/{script}', f'--date={date}', f'--tag={tag}']
    print(f'Command: {" ".join(cmd)}')

    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent.parent)
    if result.returncode == 0:
        print(f'✓ Phase {phase_id} completed')
        return True
    else:
        print(f'✗ Phase {phase_id} failed (exit {result.returncode})')
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', type=str, required=True,
                        help='Date in YYYY-MM-DD format (e.g., 2026-10-29)')
    parser.add_argument('--tag', type=str, default='default',
                        help='Signal tag (default: default)')
    parser.add_argument('--from', dest='from_phase', type=str, default='0a',
                        choices=[p[0] for p in PHASES], help='Start from phase')
    parser.add_argument('--only', dest='only_phase', type=str,
                        choices=[p[0] for p in PHASES], help='Run only this phase')
    parser.add_argument('--check', action='store_true', help='Check prerequisites only')
    args = parser.parse_args()

    run_id = f'{args.date}_{args.tag}'
    print('=' * 60)
    print('PIPELINE RUNNER')
    print('=' * 60)
    print(f'Run ID: {run_id}')

    # Determine phases to run
    if args.only_phase:
        phases_to_run = [args.only_phase]
    else:
        start_idx = next(i for i, (pid, _, _) in enumerate(PHASES) if pid == args.from_phase)
        phases_to_run = [pid for pid, _, _ in PHASES[start_idx:]]

    print(f'Phases: {phases_to_run}')

    if args.check:
        print('\nPrerequisite check:')
        for pid in phases_to_run:
            missing = check_prereqs(pid, run_id)
            status = '✓' if not missing else '✗'
            print(f'  {pid}: {status}')
            for m in missing:
                print(f'    - {m}')
        return

    # Run phases
    success = True
    for pid in phases_to_run:
        script = next(s for p, s, _ in PHASES if p == pid)
        if not run_phase(pid, script, args.date, args.tag):
            success = False
            break

    print('\n' + '=' * 60)
    if success:
        print('✓ ALL PHASES COMPLETED')
        print(f'Results in data/clean/equities/US/SP500/daily/backtest/{run_id}/')
    else:
        print('✗ PIPELINE FAILED')
    print('=' * 60)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()