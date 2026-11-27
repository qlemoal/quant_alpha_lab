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

HOW TO READ THE RESULT (the console prints it in this order)
    1. The HOLDOUT column, not the DEVELOPMENT one.
    2. RETURN PER YEAR: before costs + trading costs = after costs. A large cost line means the book trades too much.
    3. The Sharpe 95% range is wide: a few years of data cannot pin a Sharpe down.
    4. YEAR BY YEAR: does the result hold every year, or does one year carry the sample?
    5. VERDICT: skill after costs, survives 2x costs, market neutral, risk model right.
    metrics.json also holds the alpha against the market (Newey-West t), hit rate, long and short legs.
    Missing from the simulation: borrow cost, financing cost, survivorship bias of the universe.

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
    out.stage('09', 'Backtest with trading costs', 'Would this book have made money after paying for its trades?')
    out.info('run', run_id)
    out.info('cost assumptions', f'${c["aum"] / 1e6:,.0f}M book. Each trade pays half the bid-ask spread '
             f'({c["spread_kwargs"].get("spread_bps")} bps), a commission ({c["commission_bps"]} bp) and a market '
             f'impact that grows with the square root of (trade size / daily volume), k = {c["impact_kappa"]}')
    out.note('not in the costs: borrow fee of the shorts, financing. A real book would do a little worse.')

    # [1/4] -----------------------------------------------------------------------------------------------------------
    out.step(1, 4, 'Read the weights and the prices')
    weights_df = pl.read_parquet(weights_file, columns=['date', 'ticker', 'weight'])
    ex_ante_vol = float(pl.read_parquet(diag_file)['ex_ante_vol'].mean())
    panel = make_price_panel(paths.ohlcv_file, paths.features_file)
    holdout_start = holdout_start_date(model_file, paths.features_file)
    out.info('rebalances', f'{weights_df["date"].n_unique()} dates, {weights_df["ticker"].n_unique()} stocks')
    out.info('prices', f'{len(panel.dates)} dates x {len(panel.tickers)} stocks')
    out.info('delisted stocks', f'{int((panel.marked & ~panel.tradable).any(axis=0).sum())} '
             '(a position in a dead stock is closed at its last price, the day after)')
    out.info('holdout starts', f'{holdout_start}: no model has seen the dates after it' if holdout_start
             else 'unknown (stage 06 reported no split)')

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
    out.step(3, 4, 'Results. Only the HOLDOUT column is an honest test',
             'the DEVELOPMENT column uses scores from models that were also trained on later data: you could not '
             'have traded it live. Use it to debug, never to judge')
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

    yearly = yearly_table(dates_all, returns_df['gross_ret'].to_numpy(), returns_df['cost'].to_numpy(), mkt_all,
                          trades_df['entry_date'].to_numpy(), trades_df['turnover'].to_numpy())
    print_yearly(yearly, holdout_start)
    judge_holdout(metrics.get('holdout'))

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

def _years_label(m: dict) -> str:
    return f'{str(m["start"])[:7]} to {str(m["end"])[:7]}'


def print_metrics_table(metrics: dict) -> None:
    '''One row per number, one column per period. Without a holdout, the whole period is shown.'''
    names = [n for n in ('pre_holdout', 'holdout') if (metrics.get(n) or {}).get('sharpe') is not None]
    if not names:
        names = ['full'] if (metrics.get('full') or {}).get('sharpe') is not None else []
    if not names:
        out.warn('no period is long enough for metrics')
        return
    header = {'full': 'WHOLE PERIOD', 'pre_holdout': 'DEVELOPMENT', 'holdout': 'HOLDOUT'}
    pct = lambda key: (lambda m: num(m.get(key), '+.1%'))
    sh = lambda key: (lambda m: num(m.get(key), '+.2f'))
    def ci(m):
        lo, hi = m.get('sharpe_ci95') or (None, None)
        return f'{num(lo, "+.1f")} to {num(hi, "+.1f")}'
    rows = [
        ('dates', _years_label),
        ('years', lambda m: f'{m["years"]:.1f}'),
        None,
        ('RETURN PER YEAR', None),
        ('  before costs', pct('gross_return_simple_ann')),
        ('  trading costs', lambda m: num(-m['cost_ann'], '+.1%')),
        ('  after costs', lambda m: num(m['gross_return_simple_ann'] - m['cost_ann'], '+.1%')),
        None,
        ('SHARPE RATIO (return / risk)', None),
        ('  before costs', sh('sharpe_gross')),
        ('  after costs', sh('sharpe')),
        ('    95% range', ci),
        ('  if costs were 2x', lambda m: num((m.get('sharpe_vs_cost_multiplier') or {}).get('2x'), '+.2f')),
        None,
        ('RISK', None),
        ('  volatility per year', lambda m: num(m.get('ann_vol'), '.1%')),
        ('  worst fall from a peak', lambda m: num(m.get('max_drawdown'), '.1%')),
        ('  market beta (target 0)', sh('market_beta')),
        ('  realised / planned vol', lambda m: num(m.get('realised_over_ex_ante_vol'), '.2f')),
        None,
        ('TRADING', None),
        ('  book traded per year', lambda m: num(m.get('annual_turnover'), '.0f') + ' times'),
    ]
    widths = [-28] + [20] * len(names)
    out.table([''] + [header[n] for n in names], [], widths)
    for r in rows:
        if r is None:
            print()
        elif r[1] is None:
            print(out.INDENT + r[0])
        else:
            out.table_row([r[0]] + [r[1](metrics[n]) for n in names], widths)
    out.note('returns per year are simple sums of daily returns, so "before costs" + "trading costs" = "after '
             'costs". The 95% range of the Sharpe ratio is Lo (2002); 3 years of data give about +/- 1.1.')


def print_yearly(yearly: list, holdout_start) -> None:
    '''One row per calendar year: does the result hold every year, or does one year carry it?'''
    hy = holdout_start.year if holdout_start is not None else None
    def part(y):
        if hy is None:
            return ''
        return 'holdout' if y > hy else ('both' if y == hy else 'development')
    print()
    print(out.INDENT + 'YEAR BY YEAR (one good year can carry a whole sample: look for a steady pattern)')
    full = [r for r in yearly if r['n_days'] >= 120]
    out.table(['year', 'part', 'after costs', 'Sharpe before', 'Sharpe after', 'costs', 'traded'],
              [[r['year'], part(r['year']), num(r['net_return'], '+.1%'), num(r['sharpe_gross'], '+.1f'),
                num(r['sharpe_net'], '+.1f'), num(-r['cost_bps'] / 1e4, '+.1%'), num(r['turnover'], '.0f') + 'x']
               for r in full], [4, -11, 11, 13, 12, 7, 7])
    n_short = len(yearly) - len(full)
    gains = sum(1 for r in full if r['net_return'] > 0)
    print(out.INDENT + f'{gains} of {len(full)} years gain after costs'
          + (f' ({n_short} partial year of less than 120 days not shown)' if n_short else ''))


def judge_holdout(m: "dict | None") -> None:
    '''The verdict, in plain words, on the holdout only. Each line says what was checked and what it means.'''
    print()
    print(out.INDENT + 'VERDICT (holdout only)')
    if not m or m.get('sharpe') is None:
        out.warn('no holdout: the only honest number is missing')
        return
    lo, hi = m['sharpe_ci95']
    out.verdict(lo > 0, (f'skill after costs: Sharpe {m["sharpe"]:+.2f}, 95% range {lo:+.1f} to {hi:+.1f}. ')
                + ('The range is above 0.' if lo > 0 else 'The range includes 0: this result could be luck.'))

    gross, net, two_x = m.get('sharpe_gross'), m['sharpe'], (m.get('sharpe_vs_cost_multiplier') or {}).get('2x')
    if gross is not None and two_x is not None:
        out.verdict(two_x > 0, f'survives higher costs: Sharpe {gross:+.2f} before costs, {net:+.2f} after, '
                    f'{two_x:+.2f} if costs were twice the assumption.')
        if two_x <= 0 and gross > 1:
            out.note(f'the signal works before costs, but the book is traded {m["annual_turnover"]:.0f} times a year, '
                     f'which costs {m["cost_ann"]:.0%} of capital a year. Trade less first (stage 08), do not look for '
                     'a better signal.')
    beta = m.get('market_beta')
    if beta is not None:
        out.verdict(abs(beta) < 0.1, f'market neutral: beta {beta:+.2f} (should be between -0.1 and +0.1).')
    ratio = m.get('realised_over_ex_ante_vol')
    if ratio is not None:
        out.verdict(0.7 <= ratio <= 1.3, f'risk model: realised volatility = {ratio:.2f} x the planned one '
                    '(should be between 0.7 and 1.3).')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Backtest the 08 weights with trading costs')
    parser.add_argument('--date', type=str, required=True, help='Run date, YYYY-MM-DD (used in the run_id)')
    parser.add_argument('--tag', type=str, default='default', help='Run tag (default: default)')
    args = parser.parse_args()
    main(args.date, args.tag)
