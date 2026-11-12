'''
===============================================
Phase 04f: Run Backtest
===============================================

QUESTION ANSWERED
    "What would this strategy have earned, after trading costs, and is the number believable?"

THE STAGE IN ONE PICTURE

    weights.parquet (04e)      ohlcv.parquet + features.parquet      model.json (04c)
    (signal date, ticker, w)   (open_adj, ADV, volatility)           (holdout length, horizon)
            |                              |                                |
            v                              v                                v
     [1] load: weights, price panel, holdout start date
            |
            v
     [2] src/portfolio/weights_backtest.py  simulate()
            signal at close t  ->  trade at open t+1  ->  hold, weights drift  ->  next trade
            costs at every trade: half-spread + square-root impact + commission
            |
            v
     [3] metrics: whole period | before holdout | HOLDOUT   (the holdout is the number that counts)
            |
            v
     [4] returns.parquet, trades.parquet, metrics.json, equity_curve.png, config.json

WHERE IS THE COMPLICATED PART?
    src/portfolio/weights_backtest.py   all the logic and conventions (read its header: they matter)
    src/portfolio/costs.py              the cost formulas
    This script only calls them, in order.

HOW TO READ THE RESULT
    1. Look at the HOLDOUT block, not at the whole period.
    2. The Sharpe interval (sharpe_ci95) is wide: a few years of data cannot pin a Sharpe down.
    3. market_beta should be close to 0 (the book is dollar-neutral).
    4. realised_over_ex_ante_vol far from 1 means the risk model is wrong.
    5. sharpe_vs_cost_multiplier: if the Sharpe collapses at 2x costs, the strategy is fragile.
    6. Missing from the simulation: borrow cost, financing cost, delisting returns, survivorship bias.

INPUTS
    {clean_base}/weights/{run_id}/weights.parquet, weights_diag.parquet
    {clean_base}/en_model/{run_id}/model.json      (optional: gives the holdout dates)
    {clean_base}/ohlcv/ohlcv.parquet               (open_adj)
    {clean_base}/features/features.parquet         (log_adv60, vol60, fwdret{h})

OUTPUTS   {clean_base}/backtest/{run_id}/
    returns.parquet     [date, gross_ret, long_ret, short_ret, cost, market_ret, net_ret]  daily
    trades.parquet      one row per rebalance: turnover, cost by component, exposures
    metrics.json        full | pre_holdout | holdout
    equity_curve.png
    config.json

USAGE
    python scripts/pipeline/04f_run_backtest.py --date 2026-11-07 --tag default
'''

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')                      # no window: the plot is saved to a file

# Make `import src...` and `import config...` work wherever the script is started from.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import polars as pl

from config.paths import get_paths, run_id_from_args
from src.portfolio.weights_backtest import make_price_panel, simulate, summarise, net_returns, plot_equity


# =============================================================================
# CONFIGURATION
# =============================================================================

BACKTEST_CONFIG = {
    # ---- ASSUMPTIONS: review them, they drive the cost estimate ----
    'aum': 1e8,                           # dollars. Impact grows with trade size / ADV, so AUM matters.
    'spread_model': 'fixed',              # 'fixed', 'power_law' or 'bucket' (see costs.estimate_spread)
    'spread_kwargs': {'spread_bps': 3.0}, # HALF-spread paid on each trade, in basis points
    'impact_kappa': 0.5,                  # square-root impact coefficient (costs.py default is 1.0)
    'commission_bps': 1.0,                # per dollar traded, in basis points
}


def holdout_start_date(model_file: Path, features_file: Path):
    '''
    First date of the 04c holdout, or None if it cannot be determined.
    Uses model.json['holdout_start'] if 04c saved it. Otherwise it applies the 04c rule:
    the last `holdout_dates` dates that have a forward-return label.
    '''
    if not model_file.exists():
        return None
    with open(model_file) as f:
        model = json.load(f)
    if model.get('holdout_start'):
        return np.datetime64(model['holdout_start'], 'D').astype(object)

    horizon, n_hold = model.get('horizon'), model.get('holdout_dates')
    fwd_col = model.get('fwd_ret_col') or f'fwdret{horizon}'
    if not n_hold:
        return None
    labelled = (
        pl.scan_parquet(features_file)
        .filter(pl.col(fwd_col).is_not_null())
        .select(pl.col('date').unique().sort())
        .collect()['date'].to_list()
    )
    return labelled[-n_hold] if len(labelled) > n_hold else None


def main(run_date: str, tag: str):
    run_id = run_id_from_args(run_date, tag)
    paths = get_paths(run_id=run_id)

    weights_file = paths.stage_dir('weights') / 'weights.parquet'
    diag_file = paths.stage_dir('weights') / 'weights_diag.parquet'
    model_file = paths.stage_dir('en_model') / 'model.json'
    out_dir = paths.stage_dir('backtest')

    for f in (weights_file, diag_file, paths.ohlcv_file, paths.features_file):
        if not f.exists():
            raise FileNotFoundError(f'Missing input: {f}')
    out_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 70)
    print('PHASE 04f: RUN BACKTEST')
    print('=' * 70)
    print(f'Run ID:  {run_id}')
    print(f'Config:  {BACKTEST_CONFIG}')
    print(f'Output:  {out_dir}')

    # ------------------------------------------------------------------
    # [1/4] Load
    # ------------------------------------------------------------------
    print('\n[1/4] Loading weights and prices...')
    weights_df = pl.read_parquet(weights_file, columns=['date', 'ticker', 'weight'])
    ex_ante_vol = float(pl.read_parquet(diag_file)['ex_ante_vol'].mean())
    panel = make_price_panel(paths.ohlcv_file, paths.features_file)
    holdout_start = holdout_start_date(model_file, paths.features_file)
    print(f'  {weights_df["date"].n_unique()} rebalance dates, {weights_df["ticker"].n_unique()} tickers')
    print(f'  price panel: {len(panel.dates)} dates x {len(panel.tickers)} tickers')
    print(f'  holdout starts: {holdout_start if holdout_start else "unknown (no split reported)"}')

    # ------------------------------------------------------------------
    # [2/4] Simulate
    # ------------------------------------------------------------------
    print('\n[2/4] Simulating (trade at the open after each signal date, costs at every trade)...')
    returns_df, trades_df = simulate(weights_df, panel, BACKTEST_CONFIG)
    returns_df = returns_df.with_columns(
        pl.Series('net_ret', net_returns(returns_df['gross_ret'].to_numpy(), returns_df['cost'].to_numpy()))
    )
    print(f'  {returns_df.height} trading days, {trades_df.height} rebalances')

    # ------------------------------------------------------------------
    # [3/4] Metrics
    # ------------------------------------------------------------------
    print('\n[3/4] Computing metrics...')
    metrics = summarise(returns_df, trades_df, holdout_start, ex_ante_vol)
    for name in ('full', 'pre_holdout', 'holdout'):
        m = metrics.get(name)
        if not m or m.get('sharpe') is None:
            continue
        ci = m['sharpe_ci95']
        print(f'\n  --- {name.upper()} ({m["start"]} to {m["end"]}, {m["years"]} years) ---')
        print(f'  Sharpe (net)       {m["sharpe"]:.2f}   95% interval [{ci[0]:.2f}, {ci[1]:.2f}]   gross {m["sharpe_gross"]:.2f}')
        print(f'  Annual return      {m["ann_return"]:.2%}   vol {m["ann_vol"]:.2%}   max drawdown {m["max_drawdown"]:.2%}')
        if m['market_beta'] is not None:
            print(f'  Market beta        {m["market_beta"]:.3f}   (about 0 expected)')
        print(f'  Annual turnover    {m["annual_turnover"]:.1f}x   cost {m["annual_cost_bps"]:.0f} bps / year')
        print(f'  Sharpe vs cost x   {m["sharpe_vs_cost_multiplier"]}')
        if m.get('realised_over_ex_ante_vol') is not None:
            print(f'  Realised / ex-ante vol  {m["realised_over_ex_ante_vol"]:.2f}   (about 1 expected)')

    # ------------------------------------------------------------------
    # [4/4] Save
    # ------------------------------------------------------------------
    print('\n[4/4] Saving...')
    returns_df.write_parquet(out_dir / 'returns.parquet')
    trades_df.write_parquet(out_dir / 'trades.parquet')
    with open(out_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    plot_equity(returns_df, out_dir / 'equity_curve.png', f'Equity curve - {run_id}', holdout_start)
    with open(out_dir / 'config.json', 'w') as f:
        json.dump({
            'run_id': run_id,
            'tag': tag,
            'backtest_config': BACKTEST_CONFIG,
            'holdout_start': holdout_start,
            'start_date': returns_df['date'][0],
            'end_date': returns_df['date'][-1],
            'n_days': returns_df.height,
            'n_rebalances': trades_df.height,
        }, f, indent=2, default=str)
    print(f'  output: {out_dir}/')
    print('\nDone. Judge the strategy on the HOLDOUT block above.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Backtest the 04e weights with trading costs')
    parser.add_argument('--date', type=str, required=True, help='Run date, YYYY-MM-DD (used in the run_id)')
    parser.add_argument('--tag', type=str, default='default', help='Run tag (default: default)')
    args = parser.parse_args()
    main(args.date, args.tag)