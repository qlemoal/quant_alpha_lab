'''
Backtest of the weights from stage 04e (used by 04f_run_backtest.py).

The script only calls these functions. All the logic and all the conventions are here.

WHAT IS SIMULATED
-----------------
A weekly rebalanced long-short book, executed at the OPEN, marked at the OPEN, with trading costs.

    close of t            open of t+1                 open of t'+1 (next rebalance + 1 day)
    signal + weights  ->  TRADE at open_adj[t+1]  ->  hold ... daily marks ...  ->  TRADE again
    (from 04e)            pay costs                   weights drift with prices

    The label of the research (fwdret) is open-to-open too, so signal, label and P&L use the same clock.

HOW ONE HOLDING PERIOD IS COMPUTED  (no loop over days, no loop over stocks)
---------------------------------------------------------------------------
    ratio[d, i] = open[d, i] / open[entry, i]            price of stock i relative to the entry price
    NAV[d]      = 1 + sum_i w_i * (ratio[d, i] - 1)       buy-and-hold: weights are NOT rebalanced inside the period
    daily gross return = NAV[d] / NAV[d-1] - 1
    Long and short contributions are the same formula restricted to w > 0 and w < 0.

    At the end of the period the weights have drifted:
        w_drift_i = w_i * ratio_i / NAV_end
    The next trade moves the book from w_drift to the new target. The cost depends on that difference.

COSTS (src/portfolio/costs.py), charged on the entry date, as a fraction of AUM:
    half-spread  +  square-root market impact  +  commission
    AUM is held constant (profits are withdrawn). ADV and volatility are those known at the close of the signal date.

CONVENTIONS AND ASSUMPTIONS (read them)
---------------------------------------
- Prices are dividend- and split-adjusted opens (open_adj), so returns are total returns.
- Cash earns nothing. No borrow cost on the short leg. No financing cost of leverage. (All of these make
  the result slightly optimistic. They are listed so that nobody forgets them.)
- A stock with no price on the entry date cannot be traded: its weight is dropped (counted in the trade log).
- A stock whose prices stop (delisting) keeps its last price: it earns 0 until the next rebalance.
  The data holds today's S&P 500 members only, so survivorship bias is still there.
- The last period is marked at the last available open. There is no liquidation cost at the end.
'''

from dataclasses import dataclass

import numpy as np
import polars as pl

from src.portfolio.costs import estimate_spread, estimate_market_impact, estimate_commission
from src.evaluation.signals.performance import sharpe_ratio, max_drawdown

TRADING_DAYS = 252


# =============================================================================
# 1. PRICE PANEL
# =============================================================================

@dataclass
class PricePanel:
    '''Wide tables, rows = dates, columns = tickers (both sorted).'''
    dates: np.ndarray      # (T,) datetime64[D]
    tickers: np.ndarray    # (N,) str
    open: np.ndarray       # (T, N) adjusted open, forward-filled, NaN before the first price of a ticker
    adv: np.ndarray        # (T, N) average dollar volume (60 days), NaN if unknown
    vol: np.ndarray        # (T, N) daily volatility of log returns (60 days), NaN if unknown


def _pivot(long: pl.DataFrame, value: str, tickers: list) -> np.ndarray:
    wide = long.pivot(on='ticker', index='date', values=value, aggregate_function='first').sort('date')
    return wide.select(tickers).to_numpy().astype(float)


def make_price_panel(ohlcv_file, features_file) -> PricePanel:
    '''
    Build the PricePanel from ohlcv.parquet (open_adj) and features.parquet (log_adv60, vol60).
    Non-positive prices become missing. Opens are forward-filled (see the delisting rule above).
    '''
    ohlcv = pl.read_parquet(ohlcv_file, columns=['date', 'ticker', 'open_adj'])
    feats = pl.read_parquet(features_file, columns=['date', 'ticker', 'log_adv60', 'vol60'])
    long = (
        ohlcv.join(feats, on=['date', 'ticker'], how='left')
        .with_columns(
            pl.when(pl.col('open_adj') > 0).then(pl.col('open_adj')).otherwise(None).alias('open_adj'),
            pl.col('log_adv60').exp().alias('adv'),
        )
    )
    tickers = sorted(long['ticker'].unique().to_list())
    dates = long.select('date').unique().sort('date')['date'].to_numpy().astype('datetime64[D]')

    # forward-fill the open of each ticker (missing days, delisting)
    wide_open = (
        long.pivot(on='ticker', index='date', values='open_adj', aggregate_function='first').sort('date')
        .select(tickers).fill_null(strategy='forward')
    )
    open_ = wide_open.to_numpy().astype(float)
    return PricePanel(
        dates=dates,
        tickers=np.asarray(tickers),
        open=open_,
        adv=_pivot(long, 'adv', tickers),
        vol=_pivot(long, 'vol60', tickers),
    )


# =============================================================================
# 2. TRADING COST FOR ONE REBALANCE
# =============================================================================

def rebalance_cost(w_prev: np.ndarray, w_new: np.ndarray, adv: np.ndarray, vol: np.ndarray,
                   cfg: dict) -> dict:
    '''
    Cost of moving the book from w_prev to w_new, as fractions of AUM.

    w_prev, w_new: (N,) weights over ALL tickers of the panel (0 = not held).
    adv, vol:      (N,) ADV in dollars and daily volatility, known at the signal date. NaN are filled with
                   the cross-sectional median (the names that trade are almost always liquid S&P 500 stocks).
    cfg:           BACKTEST_CONFIG of the script.

    Returns dict: spread, impact, commission, total (fractions of AUM), turnover (one-way, sum|dw| / 2).
    '''
    dw = w_new - w_prev
    traded = np.abs(dw) > 1e-12
    out = {'spread': 0.0, 'impact': 0.0, 'commission': 0.0, 'total': 0.0, 'turnover': float(0.5 * np.abs(dw).sum())}
    if not traded.any():
        return out

    def fill(x, fallback):
        x = x[traded].copy()
        ok = np.isfinite(x) & (x > 0)
        x[~ok] = np.median(x[ok]) if ok.any() else fallback
        return x

    adv_t = fill(adv, 1e7)       # fallback: 10 million dollars a day
    vol_t = fill(vol, 0.02)      # fallback: 2% a day
    dw_t = dw[traded]
    aum = cfg['aum']

    spread_bps = estimate_spread(adv_t, vol_t, model=cfg['spread_model'], **cfg['spread_kwargs'])
    spread = float(np.sum(spread_bps / 1e4 * np.abs(dw_t) * aum) / aum)
    impact = float(np.sum(estimate_market_impact(dw_t, adv_t, vol_t, aum, model='sqrt',
                                                 kappa=cfg['impact_kappa'])) / aum)
    commission = float(np.sum(estimate_commission(dw_t, aum, commission_bps=cfg['commission_bps'])) / aum)
    out.update(spread=spread, impact=impact, commission=commission, total=spread + impact + commission)
    return out


# =============================================================================
# 3. THE SIMULATION
# =============================================================================

def simulate(weights_df: pl.DataFrame, panel: PricePanel, cfg: dict) -> "tuple[pl.DataFrame, pl.DataFrame]":
    '''
    Run the backtest. weights_df has columns [date, ticker, weight] (04e output, date = signal date).

    Returns (returns_df, trades_df).

    returns_df: one row per trading day from the first entry date. Value of a row = return from the
        previous open to the open of `date`.
        gross_ret   return of the book before costs
        long_ret    contribution of the long leg   (long_ret + short_ret = gross_ret)
        short_ret   contribution of the short leg
        cost        trading cost paid at the open of `date` (fraction of AUM, 0 on non-trade days)
        market_ret  equal-weight average open-to-open return of all stocks (for the beta check)
    trades_df: one row per rebalance (see the keys below).
    '''
    dates, tickers = panel.dates, panel.tickers
    T, N = panel.open.shape
    col_of = {t: i for i, t in enumerate(tickers)}

    # --- 1. entry index of every rebalance: first trading day AFTER the signal date -----------------
    book = {}
    for key, g in weights_df.sort('date').partition_by(['date'], as_dict=True, maintain_order=True).items():
        d = key[0] if isinstance(key, tuple) else key
        book[d] = g
    entries = []
    for d in sorted(book):
        e = int(np.searchsorted(dates, np.datetime64(d, 'D'), side='right'))
        if e < T:                                  # a signal on the very last date cannot be traded
            entries.append((d, e))
    if not entries:
        raise ValueError('No rebalance date can be traded: the weights end on the last price date.')
    entry_idx = [e for _, e in entries]
    if any(b <= a for a, b in zip(entry_idx, entry_idx[1:])):
        raise ValueError('Two rebalance dates map to the same entry date.')

    # --- 2. loop over holding periods (one iteration per rebalance) ---------------------------------
    gross = np.zeros(T)
    long_c = np.zeros(T)
    short_c = np.zeros(T)
    cost = np.zeros(T)
    trades = []
    w_prev = np.zeros(N)                           # weights just before the trade (drifted)

    for k, (sig_date, e) in enumerate(entries):
        g = book[sig_date]
        cols = np.array([col_of.get(t, -1) for t in g['ticker'].to_list()])
        w_all = g['weight'].to_numpy()
        known = cols >= 0
        tradable = known.copy()
        tradable[known] = np.isfinite(panel.open[e, cols[known]])        # needs a price on the entry date
        n_untradable = int((~tradable).sum())
        idx, w = cols[tradable], w_all[tradable]

        w_new = np.zeros(N)
        w_new[idx] = w

        # cost of the trade at the open of e. ADV and vol are those of the signal date (row e - 1).
        c = rebalance_cost(w_prev, w_new, panel.adv[e - 1], panel.vol[e - 1], cfg)
        cost[e] += c['total']

        trades.append({
            'signal_date': sig_date,
            'entry_date': dates[e].astype(object),
            'turnover': c['turnover'],
            'cost_spread': c['spread'], 'cost_impact': c['impact'], 'cost_commission': c['commission'],
            'cost_total': c['total'],
            'gross_exposure': float(np.abs(w).sum()),
            'net_exposure': float(w.sum()),
            'n_positions': int((np.abs(w) > 1e-12).sum()),
            'n_untradable': n_untradable,
        })

        # holding period: rows e .. end (inclusive). The next trade happens at the open of row `end`.
        end = entry_idx[k + 1] if k + 1 < len(entries) else T - 1
        m = end - e
        if m <= 0:
            w_prev = w_new
            continue

        ratio = panel.open[e:end + 1][:, idx] / panel.open[e, idx]        # (m + 1, n_held)
        excess = ratio - 1.0
        is_long, is_short = w > 0, w < 0
        d_long = excess[:, is_long] @ w[is_long]                          # (m + 1,)
        d_short = excess[:, is_short] @ w[is_short]
        nav = 1.0 + d_long + d_short                                      # buy-and-hold NAV, starts at 1

        rows = slice(e + 1, end + 1)
        gross[rows] = nav[1:] / nav[:-1] - 1.0
        long_c[rows] = np.diff(d_long) / nav[:-1]
        short_c[rows] = np.diff(d_short) / nav[:-1]

        # weights after drift, input of the next trade
        w_prev = np.zeros(N)
        w_prev[idx] = w * ratio[-1] / nav[-1]

    # --- 3. assemble the daily table -----------------------------------------------------------------
    first = entry_idx[0]
    with np.errstate(invalid='ignore', divide='ignore'):
        daily = panel.open[1:] / panel.open[:-1] - 1.0
    market = np.full(T, np.nan)
    finite_rows = np.isfinite(daily).any(axis=1)
    market[1:][finite_rows] = np.nanmean(daily[finite_rows], axis=1)

    returns_df = pl.DataFrame({
        'date': pl.Series('date', dates[first:].astype(object).tolist(), dtype=pl.Date),
        'gross_ret': gross[first:],
        'long_ret': long_c[first:],
        'short_ret': short_c[first:],
        'cost': cost[first:],
        'market_ret': market[first:],
    })
    return returns_df, pl.DataFrame(trades)


# =============================================================================
# 4. METRICS
# =============================================================================

def net_returns(gross_ret: np.ndarray, cost: np.ndarray, cost_multiplier: float = 1.0) -> np.ndarray:
    '''Net daily return: the cost is paid at the open, after the return of the day is earned.'''
    return (1.0 + gross_ret) * (1.0 - cost_multiplier * cost) - 1.0


def _num(x):
    '''JSON-safe number: None for missing or infinite values.'''
    if x is None:
        return None
    x = float(x)
    return x if np.isfinite(x) else None


def period_metrics(df: pl.DataFrame, trades: pl.DataFrame, ex_ante_vol: "float | None" = None,
                   cost_multipliers=(0.0, 1.0, 2.0, 3.0)) -> dict:
    '''
    Performance numbers of ONE period (a slice of returns_df, with the trades of the same slice).
    Daily returns, annualised with 252. The holding periods do not overlap, so the corrected
    Sharpe and drawdown functions of src/evaluation/signals/performance.py apply with horizon = 1.
    '''
    n = df.height
    if n < 20:
        return {'n_days': n, 'note': 'too short for metrics'}

    gross = df['gross_ret'].to_numpy()
    cost = df['cost'].to_numpy()
    net = net_returns(gross, cost)
    years = n / TRADING_DAYS

    sr = sharpe_ratio(net)
    ann_ret = float(np.prod(1.0 + net) ** (1.0 / years) - 1.0)
    mdd = max_drawdown(net)

    # market beta (a dollar-neutral book should be close to 0)
    mkt = df['market_ret'].to_numpy()
    ok = np.isfinite(mkt) & np.isfinite(net)
    beta = corr = None
    if ok.sum() > 20 and np.var(mkt[ok]) > 0:
        beta = float(np.cov(net[ok], mkt[ok])[0, 1] / np.var(mkt[ok], ddof=1))
        corr = float(np.corrcoef(net[ok], mkt[ok])[0, 1])

    n_reb = trades.height
    out = {
        'start': str(df['date'][0]), 'end': str(df['date'][-1]), 'n_days': n, 'years': round(years, 2),
        'ann_return': ann_ret,
        'ann_vol': float(net.std(ddof=1) * np.sqrt(TRADING_DAYS)),
        'sharpe': sr,
        'sharpe_ci95': ([sr - 1.96 * np.sqrt((1 + 0.5 * sr ** 2) / years),
                         sr + 1.96 * np.sqrt((1 + 0.5 * sr ** 2) / years)] if sr is not None else None),
        'sharpe_gross': sharpe_ratio(gross),
        'max_drawdown': mdd,
        'calmar': (ann_ret / abs(mdd)) if mdd else None,
        'hit_rate': float((net > 0).mean()),
        'long_contribution_ann': float(df['long_ret'].mean() * TRADING_DAYS),
        'short_contribution_ann': float(df['short_ret'].mean() * TRADING_DAYS),
        'market_beta': beta,
        'market_corr': corr,
        'n_rebalances': n_reb,
        'avg_turnover_per_rebalance': float(trades['turnover'].mean()) if n_reb else None,
        'annual_turnover': float(trades['turnover'].sum() / years) if n_reb else None,
        'annual_cost_bps': float(cost.sum() / years * 1e4),
        'avg_gross_exposure': float(trades['gross_exposure'].mean()) if n_reb else None,
        'avg_untradable_names': float(trades['n_untradable'].mean()) if n_reb else None,
        'sharpe_vs_cost_multiplier': {f'{m:g}x': sharpe_ratio(net_returns(gross, cost, m)) for m in cost_multipliers},
    }
    if ex_ante_vol:
        out['ex_ante_vol'] = ex_ante_vol
        out['realised_over_ex_ante_vol'] = out['ann_vol'] / ex_ante_vol
    return {k: (_num(v) if isinstance(v, (int, float, np.floating)) and not isinstance(v, bool) else v)
            for k, v in out.items()}


def summarise(returns_df: pl.DataFrame, trades_df: pl.DataFrame, holdout_start=None,
              ex_ante_vol: "float | None" = None) -> dict:
    '''
    Metrics for the whole period, and, if holdout_start (a date) is given, separately for
        pre_holdout: before it (out-of-fold scores, hyper-parameters tuned on these dates)
        holdout:     from it on (scores of a model that never saw these dates). THE NUMBER THAT COUNTS.
    A trade belongs to a period by its entry date.
    '''
    result = {'full': period_metrics(returns_df, trades_df, ex_ante_vol)}
    if holdout_start is not None:
        for name, cond_r, cond_t in (
            ('pre_holdout', pl.col('date') < holdout_start, pl.col('entry_date') < holdout_start),
            ('holdout', pl.col('date') >= holdout_start, pl.col('entry_date') >= holdout_start),
        ):
            result[name] = period_metrics(returns_df.filter(cond_r), trades_df.filter(cond_t), ex_ante_vol)
    return result


# =============================================================================
# 5. PLOT
# =============================================================================

def plot_equity(returns_df: pl.DataFrame, path, title: str, holdout_start=None) -> None:
    '''Equity curve (net, gross, equal-weight market) and drawdown. Saved as PNG.'''
    import matplotlib.pyplot as plt

    dates = returns_df['date'].to_list()
    gross = returns_df['gross_ret'].to_numpy()
    net = net_returns(gross, returns_df['cost'].to_numpy())
    mkt = np.nan_to_num(returns_df['market_ret'].to_numpy())
    eq = np.cumprod(1 + net)
    dd = eq / np.maximum.accumulate(eq) - 1

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    ax1.plot(dates, eq, label='net of costs', lw=1.5)
    ax1.plot(dates, np.cumprod(1 + gross), label='gross (before costs)', lw=1, alpha=0.6)
    ax1.plot(dates, np.cumprod(1 + mkt), label='equal-weight market', lw=1, alpha=0.5)
    ax1.set_yscale('log')
    ax1.set_ylabel('growth of 1 (log scale)')
    ax1.set_title(title)
    ax2.fill_between(dates, dd, 0, color='tab:red', alpha=0.4)
    ax2.set_ylabel('drawdown (net)')
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
        if holdout_start is not None:
            ax.axvline(holdout_start, color='k', ls='--', lw=1)
    if holdout_start is not None:
        ax1.text(holdout_start, ax1.get_ylim()[0], ' holdout start', va='bottom', fontsize=9)
    ax1.legend(loc='upper left')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)