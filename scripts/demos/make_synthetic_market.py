#!/usr/bin/env python
'''
===============================================
Make the synthetic market (a fake dataset with a known truth)
===============================================

QUESTION ANSWERED
    "How do I get a dataset that every machine can build identically, and in which I know what is predictable?"

WHAT IT DOES
    src/data/synthetic.py simulates 300 stocks over 10 years with planted effects (read its header).
    This script writes them as yfinance-style CSV files into the raw folder of the dataset
    config/dataset_synthetic.json. Stages 02 to 09 then run on it without any change.

    simulate (src/data/synthetic.py) --> data/raw/synthetic/equities/US/SYNTH/daily/ohlcv/S000.csv ... + truth.json

ALWAYS SET THE DATASET FIRST (so the real data are never touched):
    export QAL_DATASET_CONFIG=config/dataset_synthetic.json        (macOS, Linux)

USAGE
    python scripts/demos/make_synthetic_market.py
    python scripts/demos/make_synthetic_market.py --n_tickers 100 --n_days 1500 --seed 7
    then:  python scripts/pipeline/02_etl_prices.py ; python scripts/pipeline/03_build_features.py ; ...
    or all at once:  python scripts/demos/run_synthetic_reference.py
'''

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config.paths import get_paths
from src.data.synthetic import SyntheticConfig, simulate_market, write_market


def main(n_tickers: int, n_days: int, seed: int):
    paths = get_paths()
    if paths.source != 'synthetic':
        raise SystemExit(
            'This script writes a fake market. The active dataset is '
            f'"{paths.source}", not "synthetic". Run first:\n'
            '    export QAL_DATASET_CONFIG=config/dataset_synthetic.json')
    cfg = SyntheticConfig(n_tickers=n_tickers, n_days=n_days, seed=seed)
    print(f'Simulating {n_tickers} stocks x {n_days} days (seed {seed})...')
    mkt = simulate_market(cfg)
    info = write_market(mkt, cfg, paths.raw_ohlcv_dir)
    print(f'  wrote {info["n_tickers"]} CSV files to {paths.raw_ohlcv_dir}')
    print(f'  digest {info["digest"]}')
    print('Next: python scripts/pipeline/02_etl_prices.py')


if __name__ == '__main__':
    d = SyntheticConfig()
    ap = argparse.ArgumentParser(description='Write the synthetic market as yfinance-style CSV files')
    ap.add_argument('--n_tickers', type=int, default=d.n_tickers)
    ap.add_argument('--n_days', type=int, default=d.n_days)
    ap.add_argument('--seed', type=int, default=d.seed)
    a = ap.parse_args()
    main(a.n_tickers, a.n_days, a.seed)
