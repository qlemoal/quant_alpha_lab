#!/usr/bin/env python
'''
===============================================
Survivorship lab: how much does a survivors-only file lie?
===============================================

QUESTION ANSWERED
    "On a market where I KNOW who died and how, what does the pipeline say when it only sees the survivors?"

THE EXPERIMENT (one synthetic market, three versions of it)

    simulate 300 stocks, 10 years, with DELISTINGS (src/data/synthetic.py, delist_rate = 0.02, risk higher after a bad year)
          |
          +--> SURV  only the stocks that never died          = the file a free source gives you
          +--> FULL  all stocks, the dead ones included       = what a survivorship-free vendor gives you
          +--> PIT   all stocks + the point-in-time rule universe (top 200 by dollar volume, re-ranked monthly)

    Each version goes through the SAME stages: 02 ETL, 02b universe + audit, 03 features, 04 to 09 (experiment).
    The three runs go in parallel (about 4 minutes on 8 cores).

WHAT YOU READ IN THE TABLE
    - audit verdict: SURV must be FAIL ("no dead ticker"). The audit catches a survivors-only file.
    - mean IC of the planted signals, and the admitted signals: does the survivors-only file change the answer?
    - Sharpe gross and net: the size of the bias in the final number.
    The DIRECTION of the bias depends on the market: here the dead stocks are the ones that fell and then crashed, so
    removing them removes the winners of a short leg. On real data with a back-filled index list the bias is
    usually the opposite for long-only and for "winners keep winning" strategies. The point of the lab is that the
    machinery MEASURES the bias. Do not read the synthetic size as a real size.

USAGE
    python scripts/demos/survivorship_lab.py                 # run the three versions and print the table
    python scripts/demos/survivorship_lab.py --compare-only  # only read existing outputs
Real data are never touched (each version has its own dataset folder).
'''

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import polars as pl

from config.paths import get_paths
from src.data.synthetic import SyntheticConfig, simulate_market, write_market

DATE, TAG = '2026-01-01', 'survlab'
PIPE = ROOT / 'scripts' / 'pipeline'
VARIANTS = ['SURV', 'FULL', 'PIT']
SIGNALS_OF_INTEREST = ['mom252_zscore_tanh', 'vol252_zscore_clip', 'log_adv60_zscore_tanh']
MARKET = SyntheticConfig(n_tickers=300, n_days=2520, seed=12345, delist_rate=0.02, delist_sensitivity=1.0)


def cfg_file(v: str) -> str:
    return f'config/dataset_synth_{v.lower()}.json'


def paths_of(v: str, run_id=None):
    return get_paths(source='synthetic', universe=f'SYNTH_{v}', run_id=run_id)


def make_data():
    print('Simulating the market (with delistings)...', flush=True)
    mkt = simulate_market(MARKET)
    print(f'  {len(mkt["tickers"])} stocks, {int(mkt["dead"].sum())} die before the end ({mkt["dead"].mean():.0%})')
    for v in VARIANTS:
        info = write_market(mkt, MARKET, paths_of(v).raw_ohlcv_dir, drop_dead=(v == 'SURV'))
        print(f'  {v}: wrote {info["n_tickers"]} ticker files')


def run_variants():
    steps = [[PIPE / '02_etl_prices.py'], [PIPE / '02b_build_universe.py'], [PIPE / '03_build_features.py'],
             [PIPE / 'run_experiment.py', '--date', DATE, '--tag', TAG]]
    procs = {}
    for v in VARIANTS:
        env = {**os.environ, 'QAL_DATASET_CONFIG': cfg_file(v)}
        log = open(ROOT / f'survlab_{v}.log', 'w')
        cmd = ' && '.join(' '.join([sys.executable, *map(str, s)]) for s in steps)
        procs[v] = (subprocess.Popen(cmd, shell=True, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT), log)
        print(f'  started {v} (log: survlab_{v}.log)', flush=True)
    failed = []
    for v, (p, log) in procs.items():
        p.wait(); log.close()
        if p.returncode != 0:
            failed.append(v)
    if failed:
        raise SystemExit(f'failed: {failed}. Read survlab_<VARIANT>.log')


def measure(v: str) -> dict:
    p = paths_of(v, run_id=f'{DATE}_{TAG}')
    audit = json.loads((p.universe_file.parent / 'universe_audit.json').read_text())
    sel = json.loads((p.stage_dir('signal_eval') / 'selected_signals.json').read_text())
    ic = pl.read_parquet(p.stage_dir('signal_eval') / 'ic_summary.parquet')
    model = json.loads((p.stage_dir('en_model') / 'model.json').read_text()) if sel['signals'] else {}
    bt = json.loads((p.stage_dir('backtest') / 'metrics.json').read_text())['full']
    out = {'tickers': audit['n_tickers'], 'dead': audit['n_dead'], 'verdict': audit['verdict'],
           'admitted': ' '.join(sorted(re.sub(r'_(zscore|rank|decile).*$', '', s) for s in sel['signals'])) or '-',
           'holdout_ic': model.get('holdout_report', {}).get('ic_mean'),
           'sharpe_gross': bt['sharpe_gross'], 'sharpe_net': bt['sharpe'], 'ann_return': bt['ann_return']}
    for s in SIGNALS_OF_INTEREST:
        r = ic.filter(pl.col('signal') == s)
        out['IC ' + re.sub(r'_(zscore|rank|decile).*$', '', s)] = float(r['mean_ic'][0]) if r.height else None
    return out


def main(compare_only: bool):
    if not compare_only:
        make_data()
        run_variants()
    rows = {v: measure(v) for v in VARIANTS}
    keys = list(rows['SURV'].keys())
    print('\n' + '=' * 92)
    print(f'{"":<18}' + ''.join(f'{v:>24}' for v in VARIANTS))
    for k in keys:
        def f(x):
            return f'{x:.4f}' if isinstance(x, float) else str(x)
        print(f'{k:<18}' + ''.join(f'{f(rows[v][k]):>24}' for v in VARIANTS))
    print('  (admitted signals are listed on the next lines because the names are long)')
    for v in VARIANTS:
        print(f'  {v}: {rows[v]["admitted"]}')
    print('=' * 92)
    print('SURV = survivors only. FULL = dead stocks included. PIT = dead stocks + point-in-time universe.')
    print('The audit verdict of SURV must be FAIL. The gap between SURV and FULL is the survivorship bias of this market.')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Survivorship lab on the synthetic market')
    ap.add_argument('--compare-only', action='store_true')
    main(ap.parse_args().compare_only)
