#!/usr/bin/env python
'''
===============================================
Reference run on the synthetic market
===============================================

QUESTION ANSWERED
    "Does my copy of the pipeline give the same answer as the reference, on a market where the truth is known?"

WHAT IT DOES (about 4 minutes)

    make_synthetic_market -> 02 ETL -> 03 features -> run_experiment (04 signals ... 09 backtest)
          |
          v
    read the outputs -> a dozen numbers -> compare with tests/golden/synthetic_reference.json
          |
          v
    table: metric, rule, measured, golden, PASS / FAIL

    Everything runs with QAL_DATASET_CONFIG=config/dataset_synthetic.json, so your REAL data are never read or written.
    The run id is 2026-01-01_reference.

WHAT TO LOOK AT
    - admitted_signals / families: the planted effects (low volatility, illiquidity, momentum) must be found.
      The reversal effect is planted but short-lived: it is expected NOT to pass the strict gates at the 20-day horizon.
    - cluster_ari: how well the clusters recover the 8 true industries (1 = perfect).
    - beta_exposure_max_abs: must be about 0 (the book is beta-neutral by construction).
    - realised_beta_full: the beta measured in the backtest. About 0.
    - sharpe_gross_full vs sharpe_net_full: the gap is the cost of trading. This is the weak point of the strategy.

USAGE
    python scripts/demos/run_synthetic_reference.py                  # run and compare
    python scripts/demos/run_synthetic_reference.py --update-golden  # run and overwrite the golden numbers (the author only)
    python scripts/demos/run_synthetic_reference.py --compare-only   # do not run, only read the existing outputs
'''

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ['QAL_DATASET_CONFIG'] = 'config/dataset_synthetic.json'      # BEFORE importing config.paths
sys.path.insert(0, str(ROOT))

import polars as pl

from config.paths import get_paths
from src.evaluation.reference import compare, adjusted_rand

DATE, TAG = '2026-01-01', 'reference'
GOLDEN = ROOT / 'tests' / 'golden' / 'synthetic_reference.json'
PIPE = ROOT / 'scripts' / 'pipeline'

STEPS = [
    [ROOT / 'scripts' / 'demos' / 'make_synthetic_market.py'],
    [PIPE / '02_etl_prices.py'],
    [PIPE / '03_build_features.py'],
    [PIPE / 'run_experiment.py', '--date', DATE, '--tag', TAG],
]


def run_all():
    for step in STEPS:
        print(f'\n>>> {step[0].name} {" ".join(map(str, step[1:]))}', flush=True)
        r = subprocess.run([sys.executable, *map(str, step)], cwd=ROOT, env=os.environ)
        if r.returncode != 0:
            raise SystemExit(f'step failed: {step[0].name}')


def measure() -> dict:
    p = get_paths(run_id=f'{DATE}_{TAG}')
    sel = json.loads((p.stage_dir('signal_eval') / 'selected_signals.json').read_text())
    model = json.loads((p.stage_dir('en_model') / 'model.json').read_text())
    risk = json.loads((p.stage_dir('risk_model') / 'summary.json').read_text())
    bt = json.loads((p.stage_dir('backtest') / 'metrics.json').read_text())
    diag = pl.read_parquet(p.stage_dir('weights') / 'weights_diag.parquet')

    # clusters of the last risk-model date against the true industries
    truth = json.loads((p.raw_ohlcv_dir / 'truth.json').read_text())['industry']
    cl = pl.read_parquet(p.stage_dir('risk_model') / 'clusters.parquet')
    last = cl.filter(pl.col('date') == cl['date'].max())
    ari = adjusted_rand(last['cluster_id'].to_numpy(), [truth[t] for t in last['ticker'].to_list()])

    full = bt['full']
    return {
        'admitted_signals': sorted(sel['signals']),
        'families': sorted(model['families']),
        'holdout_ic': model['holdout_report']['ic_mean'],
        'avg_n_clusters': risk['avg_n_clusters'],
        'cluster_ari': ari,
        'market_mode_removed_share': risk['market_mode_removed_share'],
        'beta_exposure_max_abs': float(diag['beta_exposure'].abs().max()),
        'realised_beta_full': full['market_beta'],
        'sharpe_gross_full': full['sharpe_gross'],
        'sharpe_net_full': full['sharpe'],
        'annual_cost_bps_full': full['annual_cost_bps'],
        'alpha_t_full': full['alpha_regression']['alpha_t'],
    }


def main(update_golden: bool, compare_only: bool):
    if not compare_only:
        run_all()
    measured = measure()
    if update_golden:
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN.write_text(json.dumps(measured, indent=2))
        print(f'\nGolden numbers written to {GOLDEN}')
        return
    golden = json.loads(GOLDEN.read_text())
    rows = compare(measured, golden)
    print('\n' + '=' * 100)
    print(f'{"metric":<28}{"rule":<12}{"measured":>34}{"golden":>34}  result')
    for r in rows:
        f = lambda v: (f'{v:.4f}' if isinstance(v, float) else str(v))
        print(f'{r["metric"]:<28}{r["rule"]:<12}{f(r["measured"]):>34}{f(r["golden"]):>34}  {"PASS" if r["ok"] else "FAIL"}')
    n_fail = sum(not r['ok'] for r in rows)
    print('=' * 100)
    print('ALL PASS' if n_fail == 0 else f'{n_fail} metric(s) differ from the reference. See the table.')
    sys.exit(1 if n_fail else 0)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Run the synthetic reference experiment and compare with the golden numbers')
    ap.add_argument('--update-golden', action='store_true')
    ap.add_argument('--compare-only', action='store_true')
    a = ap.parse_args()
    main(a.update_golden, a.compare_only)
