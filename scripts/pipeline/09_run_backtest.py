#!/usr/bin/env python
'''
===============================================
Phase 09: Run Backtest
===============================================

QUESTION ANSWERED
    "What would this strategy have earned, after trading costs, and is the number believable?"

THE STAGE IN ONE PICTURE

    weights.parquet (08)      ohlcv.parquet + features.parquet      model.json (06)
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
    3. market_beta should be close to 0. The ALPHA line gives the return left after the market exposure, with a
       Newey-West t-stat. A good Sharpe with a small alpha t-stat is market exposure, not skill.
       The YEARLY table shows whether the result holds in every year or one year carries the sample.
    4. realised_over_ex_ante_vol far from 1 means the risk model is wrong.
    5. sharpe_vs_cost_multiplier: if the Sharpe collapses at 2x costs, the strategy is fragile.
    6. Missing from the simulation: borrow cost, financing cost, delisting returns, survivorship bias.

INPUTS
    {clean_base}/runs/{run_id}/weights/weights.parquet, weights_diag.parquet
    {clean_base}/runs/{run_id}/en_model/model.json      (optional: gives the holdout dates)
    {clean_base}/ohlcv/ohlcv.parquet               (open_adj)
    {clean_base}/features/features.parquet         (log_adv60, vol60, fwdret{h})

OUTPUTS   {clean_base}/runs/{run_id}/backtest/
    returns.parquet     [date, gross_ret, long_ret, short_ret, cost, market_ret, net_ret]  daily
    trades.parquet      one row per rebalance: turnover, cost by component, exposures
    metrics.json        full | pre_holdout | holdout   (each with an 'alpha_regression' block)
    yearly.json         one row per calendar year: return, vol, Sharpe, beta, cost, turnover
    equity_curve.png
    config.json

USAGE
    python scripts/pipeline/09_run_backtest.py --date 2026-11-07 --tag default
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
from src.evaluation.portfolio.attribution import alpha_regression, yearly_table
from src.utils import console as out
from src.utils.console import num


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
    First date of the 06 holdout, or None if it cannot be determined.
    Uses model.json['holdout_start'] if 06 saved it. Otherwise it applies the 06 rule:
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

    c = BACKTEST_CONFIG
    out.stage('09', 'Run the backtest', 'What would the book have earned, after trading costs, '
              'and what is the honest number?')
    out.info('run', run_id)
    out.info('costs (assumptions)', f'AUM ${c["aum"]:,.0f}, half-spread {c["spread_kwargs"].get("spread_bps")} bps, '
             f'square-root impact k = {c["impact_kappa"]}, commission {c["commission_bps"]} bp')
    out.note('not modelled: borrow cost of shorts, financing of leverage. Both make the result optimistic.')

    # [1/4] -----------------------------------------------------------------------------------------------------------
    out.step(1, 4, 'Read the weights and the prices')
    weights_df = pl.read_parquet(weights_file, columns=['date', 'ticker', 'weight'])
    ex_ante_vol = float(pl.read_parquet(diag_file)['ex_ante_vol'].mean())
    panel = make_price_panel(paths.ohlcv_file, paths.features_file)
    holdout_start = holdout_start_date(model_file, paths.features_file)
    out.info('rebalances', f'{weights_df["date"].n_unique()} dates, {weights_df["ticker"].n_unique()} stocks')
    out.info('prices', f'{len(panel.dates)} dates x {len(panel.tickers)} stocks')
    out.info('dead stocks', f'{int((panel.marked & ~panel.tradable).any(axis=0).sum())} '
             '(their last close, with the delisting return, is booked the day after their last row)')
    out.info('holdout starts', holdout_start or 'unknown (no split reported by 06)')

    # [2/4] -----------------------------------------------------------------------------------------------------------
    out.step(2, 4, 'Simulate the book day by day',
             'trade at the open after each signal date, hold (weights drift with prices), pay costs on each trade; '
             'a stock with no real price that day cannot be traded')
    returns_df, trades_df = simulate(weights_df, panel, BACKTEST_CONFIG)
    returns_df = returns_df.with_columns(
        pl.Series('net_ret', net_returns(returns_df['gross_ret'].to_numpy(), returns_df['cost'].to_numpy()))
    )
    out.info('simulated', f'{returns_df.height} trading days, {trades_df.height} rebalances')
    n_untr = int(trades_df['n_untradable'].sum())
    if n_untr:
        out.note(f'{n_untr} target positions could not be traded (no real price on the entry day) and were dropped')

    # [3/4] -----------------------------------------------------------------------------------------------------------
    out.step(3, 4, 'Performance by period: judge the strategy on the HOLDOUT only',
             'before the holdout the scores are out-of-fold, but each comes from models trained on blocks BEFORE '
             'AND AFTER its date: that is not what you could have traded live, so it is a development number')
    metrics = summarise(returns_df, trades_df, holdout_start, ex_ante_vol)
    net_all = returns_df['net_ret'].to_numpy()
    mkt_all = returns_df['market_ret'].to_numpy()
    dates_all = returns_df['date'].to_numpy()
    for name, m in metrics.items():
        if holdout_start is not None and name == 'holdout':
            sel = dates_all >= np.datetime64(holdout_start)
        elif holdout_start is not None and name == 'pre_holdout':
            sel = dates_all < np.datetime64(holdout_start)
        else:
            sel = np.ones(len(dates_all), dtype=bool)
        m['alpha_regression'] = alpha_regression(net_all[sel], mkt_all[sel])
    print_metrics_table(metrics)
    judge_holdout(metrics.get('holdout'))

    yearly = yearly_table(dates_all, returns_df['gross_ret'].to_numpy(), returns_df['cost'].to_numpy(), mkt_all,
                          trades_df['entry_date'].to_numpy(), trades_df['turnover'].to_numpy())
    print()
    out.info('year by year', '(net of costs; one good year can carry a whole sample)')
    f = lambda v, spec: num(v, spec)
    out.table(['year', 'days', 'return', 'vol', 'Sharpe', 'gross Sh', 'beta', 'cost bps', 'turnover'],
              [[r['year'], r['n_days'], f(r['net_return'], '+.1%'), f(r['ann_vol'], '.1%'), f(r['sharpe_net'], '+.2f'),
                f(r['sharpe_gross'], '+.2f'), f(r['beta'], '+.2f'), f(r['cost_bps'], '.0f'), f(r['turnover'], '.1f')]
               for r in yearly], [4, 5, 7, 6, 7, 8, 6, 8, 8])
    out.info('positive years', f'{sum(1 for r in yearly if r["net_return"] > 0)} of {len(yearly)}')

    # [4/4] -----------------------------------------------------------------------------------------------------------
    out.step(4, 4, 'Save')
    returns_df.write_parquet(out_dir / 'returns.parquet')
    trades_df.write_parquet(out_dir / 'trades.parquet')
    with open(out_dir / 'metrics.json', 'w') as fh:
        json.dump(metrics, fh, indent=2, default=str)
    with open(out_dir / 'yearly.json', 'w') as fh:
        json.dump(yearly, fh, indent=2, default=str)
    plot_equity(returns_df, out_dir / 'equity_curve.png', f'Equity curve - {run_id}', holdout_start)
    with open(out_dir / 'config.json', 'w') as fh:
        json.dump({
            'run_id': run_id, 'tag': tag, 'backtest_config': BACKTEST_CONFIG, 'holdout_start': holdout_start,
            'start_date': returns_df['date'][0], 'end_date': returns_df['date'][-1],
            'n_days': returns_df.height, 'n_rebalances': trades_df.height,
        }, fh, indent=2, default=str)
    out.done({'metrics': out_dir / 'metrics.json', 'equity curve': out_dir / 'equity_curve.png'})


# =============================================================================
# PRINTING
# =============================================================================

def print_metrics_table(metrics: dict) -> None:
    '''One row per number, one column per period (full, pre-holdout, holdout).'''
    names = [n for n in ('full', 'pre_holdout', 'holdout') if metrics.get(n) and metrics[n].get('sharpe') is not None]
    if not names:
        out.warn('no period is long enough for metrics')
        return
    def cell(m, key, spec):
        return num(m.get(key), spec)
    def ci(m):
        lo, hi = m.get('sharpe_ci95') or (None, None)
        return f'[{num(lo, "+.1f")}, {num(hi, "+.1f")}]'
    def alpha(m):
        a = m.get('alpha_regression') or {}
        return f'{num(a.get("alpha_ann"), "+.1%")} (t {num(a.get("alpha_t"), "+.1f")})'
    rows = [
        ('period', lambda m: f'{m["start"]}..{str(m["end"])[2:]}'),
        ('years', lambda m: m['years']),
        ('Sharpe, net of costs', lambda m: cell(m, 'sharpe', '+.2f')),
        ('  95% interval', ci),
        ('Sharpe, before costs', lambda m: cell(m, 'sharpe_gross', '+.2f')),
        ('Sharpe at 2x costs', lambda m: num((m.get('sharpe_vs_cost_multiplier') or {}).get('2x'), '+.2f')),
        ('annual return, net', lambda m: cell(m, 'ann_return', '+.1%')),
        ('annual volatility', lambda m: cell(m, 'ann_vol', '.1%')),
        ('max drawdown', lambda m: cell(m, 'max_drawdown', '.1%')),
        ('market beta', lambda m: cell(m, 'market_beta', '+.2f')),
        ('alpha per year (NW t)', alpha),
        ('turnover per year', lambda m: cell(m, 'annual_turnover', '.1f') + 'x'),
        ('cost per year (bps)', lambda m: cell(m, 'annual_cost_bps', '.0f')),
        ('realised / ex-ante vol', lambda m: cell(m, 'realised_over_ex_ante_vol', '.2f')),
    ]
    header = {'full': 'FULL', 'pre_holdout': 'BEFORE HOLDOUT', 'holdout': 'HOLDOUT'}
    out.table([''] + [header[n] for n in names], [[label] + [fn(metrics[n]) for n in names] for label, fn in rows],
              [-24] + [22] * len(names))
    out.note('Sharpe interval: Lo (2002). Alpha: return left after the market exposure, Newey-West t. '
             'Realised / ex-ante vol near 1 means the risk model sized the book right.')


def judge_holdout(m: "dict | None") -> None:
    '''Plain-language verdicts on the holdout numbers.'''
    print()
    if not m or m.get('sharpe') is None:
        out.warn('no holdout block: the only clean number is missing')
        return
    lo, hi = m['sharpe_ci95']
    out.verdict(lo > 0, f'holdout net Sharpe {m["sharpe"]:+.2f}: its 95% interval [{lo:+.1f}, {hi:+.1f}] '
                + ('excludes 0' if lo > 0 else 'includes 0, so the holdout cannot tell this strategy from no skill'))
    two_x = (m.get('sharpe_vs_cost_multiplier') or {}).get('2x')
    if two_x is not None:
        out.verdict(two_x > 0, f'at 2x the assumed costs the net Sharpe is {two_x:+.2f}'
                    + ('' if two_x > 0 else ': the strategy depends on the cost assumptions'))
    beta = m.get('market_beta')
    if beta is not None:
        out.verdict(abs(beta) < 0.1, f'realised market beta {beta:+.2f} (a beta-neutral book should be near 0)')
    ratio = m.get('realised_over_ex_ante_vol')
    if ratio is not None and not (0.7 <= ratio <= 1.3):
        out.warn(f'realised volatility is {ratio:.2f} x the ex-ante one: the risk model misjudges the book')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Backtest the 08 weights with trading costs')
    parser.add_argument('--date', type=str, required=True, help='Run date, YYYY-MM-DD (used in the run_id)')
    parser.add_argument('--tag', type=str, default='default', help='Run tag (default: default)')
    args = parser.parse_args()
    main(args.date, args.tag)
