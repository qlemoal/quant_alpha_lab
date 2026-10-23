#!/usr/bin/env python
'''
===============================================
Run Full Pipeline (or individual phases)
===============================================

Sequentially runs phases 0a -> 0b -> 0c -> 1 -> 2 -> 3.
Each phase is a separate script you can also run manually.

Usage:
  python scripts/pipeline/04_run_all.py --tag default           # Run all
  python scripts/pipeline/04_run_all.py --tag default --from 2  # Run from phase 2
  python scripts/pipeline/04_run_all.py --tag default --only 1  # Run only phase 1
  python scripts/pipeline/04_run_all.py --tag default --check   # Check prerequisites only
'''

import argparse
import subprocess
import sys
from pathlib import Path

PHASES = [
    ('0a', '04a_generate_signals.py', 'Generate signals (make_signal)'),
    ('0b', '04b_evaluate_signals.py', 'Evaluate signals (IC, FDR, decay)'),
    ('0c', '04c_fit_combiner.py', 'Fit Elastic Net combiner'),
    ('1',  '04d_estimate_risk_model.py', 'Estimate risk model (cov + clusters)'),
    ('2',  '04e_construct_weights.py', 'Construct portfolio weights'),
    ('3',  '04f_run_backtest.py', 'Run backtest with costs'),
]

PREREQS = {
    '0a': ['data/processed/features.parquet', 'data/processed/prices.parquet'],
    '0b': ['data/processed/signals_{tag}.parquet'],
    '0c': ['data/processed/signals_{tag}.parquet', 'data/exports/signal_eval_{tag}/selected_signals.json'],
    '1':  ['data/processed/en_scores_{tag}.parquet', 'data/processed/prices.parquet', 'data/processed/features.parquet'],
    '2':  ['data/exports/risk_model_{tag}/cov.parquet', 'data/exports/risk_model_{tag}/clusters.parquet',
           'data/exports/risk_model_{tag}/summary.json', 'data/processed/en_scores_{tag}.parquet'],
    '3':  ['data/exports/weights_{tag}.parquet', 'data/processed/prices.parquet', 'data/processed/features.parquet'],
}


def check_prereqs(phase_id, tag):
    missing = []
    for pattern in PREREQS.get(phase_id, []):
        f = Path(pattern.format(tag=tag))
        if not f.exists():
            missing.append(str(f))
    return missing


def run_phase(phase_id, script, tag):
    print(f'\n{"="*60}')
    print(f'RUNNING PHASE {phase_id}: {script}')
    print(f'{"="*60}')

    missing = check_prereqs(phase_id, tag)
    if missing:
        print(f'✗ Prerequisites not met:')
        for m in missing:
            print(f'  - {m}')
        return False

    cmd = [sys.executable, f'scripts/pipeline/{script}', f'--signal-tag={tag}']
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
    parser.add_argument('--tag', type=str, default='default', help='Signal tag')
    parser.add_argument('--from', dest='from_phase', type=str, default='0a',
                        choices=[p[0] for p in PHASES], help='Start from phase')
    parser.add_argument('--only', dest='only_phase', type=str,
                        choices=[p[0] for p in PHASES], help='Run only this phase')
    parser.add_argument('--check', action='store_true', help='Check prerequisites only')
    args = parser.parse_args()

    print('=' * 60)
    print('PIPELINE RUNNER')
    print('=' * 60)
    print(f'Tag: {args.tag}')

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
            missing = check_prereqs(pid, args.tag)
            status = '✓' if not missing else '✗'
            print(f'  {pid}: {status}')
            for m in missing:
                print(f'    - {m}')
        return

    # Run phases
    success = True
    for pid in phases_to_run:
        script = next(s for p, s, _ in PHASES if p == pid)
        if not run_phase(pid, script, args.tag):
            success = False
            break

    print('\n' + '=' * 60)
    if success:
        print('✓ ALL PHASES COMPLETED')
        print(f'Results in data/exports/backtest_{args.tag}/')
    else:
        print('✗ PIPELINE FAILED')
    print('=' * 60)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()