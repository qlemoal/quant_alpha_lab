#!/usr/bin/env python
'''
Master Pipeline: Run All Phases
================================

Orchestrates the full portfolio construction and backtest pipeline.
Can run all phases or individual phases.

Usage:
  python scripts/pipeline/04_run_pipeline.py           # Run all phases
  python scripts/pipeline/04_run_pipeline.py --phase 1 # Run only Phase 1
  python scripts/pipeline/04_run_pipeline.py --phase 2 # Run only Phase 2
  python scripts/pipeline/04_run_pipeline.py --phase 3 # Run only Phase 3

Phases:
  0: Generate EN scores (elastic_net_combiner)
  1: Estimate risk model (covariance + clustering)
  2: Construct portfolio weights
  3: Run backtest with costs

Each phase saves intermediate results to data/exports/ for inspection.
'''

import os
import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
SCRIPTS_DIR = PROJECT_ROOT / 'scripts' / 'pipeline'

PHASES = {
    0: ('04a_generate_en_scores.py', 'Generate Elastic Net Scores'),
    1: ('04b_estimate_risk_model.py', 'Estimate Risk Model (Covariance + Clustering)'),
    2: ('04c_construct_portfolio.py', 'Construct Portfolio Weights'),
    3: ('04d_run_backtest.py', 'Run Backtest with Costs'),
}


def run_script(script_name, description):
    '''Run a pipeline script and return success status'''
    script_path = SCRIPTS_DIR / script_name
    if not script_path.exists():
        print(f'✗ Script not found: {script_path}')
        return False

    print(f'\n{"="*70}')
    print(f'RUNNING: {description}')
    print(f'SCRIPT:  {script_name}')
    print(f'{"="*70}')

    # Pass through environment variables
    env = os.environ.copy()

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            env=env,
            cwd=PROJECT_ROOT,
            check=False,
        )
        if result.returncode == 0:
            print(f'\n✓ {description} completed successfully')
            return True
        else:
            print(f'\n✗ {description} failed with exit code {result.returncode}')
            return False
    except Exception as e:
        print(f'\n✗ {description} failed with exception: {e}')
        return False


def check_prerequisites(phase):
    '''Check if prerequisite files exist for a given phase'''
    exports = PROJECT_ROOT / 'data' / 'exports'
    processed = PROJECT_ROOT / 'data' / 'processed'

    checks = {
        0: [
            processed / 'signals.parquet',
            processed / 'prices.parquet',
            processed / 'features.parquet',
        ],
        1: [
            processed / 'en_scores.parquet',  # or signals.parquet
            processed / 'prices.parquet',
            processed / 'features.parquet',
        ],
        2: [
            exports / 'risk_model' / 'covariance.parquet',
            exports / 'risk_model' / 'clusters.parquet',
            exports / 'risk_model' / 'cluster_summary.json',
            processed / 'en_scores.parquet',
            processed / 'prices.parquet',
        ],
        3: [
            exports / 'weights_pre_backtest.parquet',
            processed / 'prices.parquet',
            processed / 'features.parquet',
        ],
    }

    missing = [f for f in checks.get(phase, []) if not f.exists()]
    return missing


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Run portfolio pipeline phases')
    parser.add_argument('--phase', type=int, choices=[0, 1, 2, 3],
                        help='Run only a specific phase (0-3)')
    parser.add_argument('--from-phase', type=int, choices=[0, 1, 2, 3], default=0,
                        help='Start from this phase (default: 0)')
    parser.add_argument('--to-phase', type=int, choices=[0, 1, 2, 3], default=3,
                        help='End at this phase (default: 3)')
    parser.add_argument('--list', action='store_true', help='List phases and exit')
    parser.add_argument('--check', action='store_true', help='Check prerequisites only')

    args = parser.parse_args()

    if args.list:
        print('Pipeline Phases:')
        for num, (script, desc) in PHASES.items():
            print(f'  {num}: {desc} ({script})')
        return

    # Determine phases to run
    if args.phase is not None:
        phases_to_run = [args.phase]
    else:
        phases_to_run = list(range(args.from_phase, args.to_phase + 1))

    print('=' * 70)
    print('PORTFOLIO PIPELINE ORCHESTRATOR')
    print('=' * 70)
    print(f'Phases to run: {phases_to_run}')
    for p in phases_to_run:
        print(f'  {p}: {PHASES[p][1]}')
    print('=' * 70)

    # Check prerequisites
    if args.check or True:
        print('\nChecking prerequisites...')
        for p in phases_to_run:
            missing = check_prerequisites(p)
            status = '✓' if not missing else '✗ MISSING'
            print(f'  Phase {p}: {status}')
            for f in missing:
                print(f'    - {f}')

    if args.check:
        return

    # Run phases
    success = True
    for p in phases_to_run:
        missing = check_prerequisites(p)
        if missing:
            print(f'\n⚠ Phase {p} prerequisites not met. Skipping.')
            print('  Run previous phases first or generate required files.')
            success = False
            break

        script, desc = PHASES[p]
        if not run_script(script, desc):
            success = False
            break

    print('\n' + '=' * 70)
    if success:
        print('✓ ALL PHASES COMPLETED SUCCESSFULLY')
        print('Results in: data/exports/')
    else:
        print('✗ PIPELINE FAILED')
    print('=' * 70)

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()