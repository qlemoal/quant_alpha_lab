'''
GOAL: answer two questions the Sharpe ratio cannot answer.

    1. Is the return ALPHA, or is it market exposure in disguise?
       Regress the daily net return of the strategy on the market return:

           net_t = alpha + beta * market_t + error_t

       beta  = exposure to the market (about 0 for a market-neutral book)
       alpha = what is left after the market exposure is removed, per day. Annualised: x 252.
       t(alpha) uses Newey-West (HAC) standard errors, so autocorrelated daily returns do not inflate it.
       A Sharpe of 1.0 with beta 0.3 can have an alpha t-stat below 1: the market did the work.

    2. Does the result hold in EVERY year, or does one good year carry the whole sample?
       yearly_table gives return, volatility, Sharpe, cost and turnover year by year.

The market is the equal-weight average return of all stocks of the universe (column market_ret of stage 09).
It is a market proxy of THIS universe, not the S&P 500 index.
Size, value and momentum factors are not removed here: that needs factor data (a later step).
'''

import numpy as np

TRADING_DAYS = 252


def _nw_lag(n: int) -> int:
    '''Classic Newey-West rule: floor(4 (n / 100)^(2/9)).'''
    return int(4 * (n / 100.0) ** (2.0 / 9.0))


def alpha_regression(net, market, max_lag: "int | None" = None) -> dict:
    '''
    OLS of the daily strategy return on the market return, with Newey-West standard errors.

    net, market: 1-D arrays of the same length (daily returns). Rows with NaN in either are dropped.
    Returns {n, beta, beta_t, alpha_daily, alpha_ann, alpha_t, r2, max_lag}.
    Returns {n, note} if there are fewer than 30 usable days or the market has no variance.
    '''
    y = np.asarray(net, dtype=float)
    x = np.asarray(market, dtype=float)
    ok = np.isfinite(y) & np.isfinite(x)
    y, x = y[ok], x[ok]
    n = len(y)
    if n < 30 or np.var(x) <= 0:
        return {'n': n, 'note': 'too short or no market variance'}

    X = np.column_stack([np.ones(n), x])
    XtX_inv = np.linalg.inv(X.T @ X)
    coef = XtX_inv @ (X.T @ y)
    resid = y - X @ coef

    L = _nw_lag(n) if max_lag is None else int(max_lag)
    L = min(L, n - 1)
    scores = X * resid[:, None]                          # (n, 2): x_t e_t
    S = scores.T @ scores                                # lag 0
    for k in range(1, L + 1):
        w = 1.0 - k / (L + 1.0)                          # Bartlett weight
        G = scores[k:].T @ scores[:-k]
        S += w * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(cov))

    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float((resid ** 2).sum()) / ss_tot if ss_tot > 0 else None
    return {
        'n': n, 'max_lag': L,
        'beta': float(coef[1]), 'beta_t': float(coef[1] / se[1]) if se[1] > 0 else None,
        'alpha_daily': float(coef[0]), 'alpha_ann': float(coef[0] * TRADING_DAYS),
        'alpha_t': float(coef[0] / se[0]) if se[0] > 0 else None,
        'r2': r2,
    }


def yearly_table(dates, gross_ret, cost, market, trade_dates=None, trade_turnover=None) -> list:
    '''
    One row per calendar year, from daily arrays.

    dates:      array of datetime64 or date objects (daily)
    gross_ret:  daily return before costs        cost: daily cost as a fraction of NAV
    market:     daily market return
    trade_dates, trade_turnover: entry date and one-way turnover of each rebalance (optional)

    Row keys: year, n_days, net_return (compounded), ann_vol, sharpe_net, sharpe_gross, beta, cost_bps, turnover.
    The Sharpe here is the plain annualised mean / std of daily returns of that year. One year is a short
    sample: its Sharpe has a standard error near 1. Read the TABLE for stability, not each number.
    '''
    d = np.asarray(dates, dtype='datetime64[D]')
    years = d.astype('datetime64[Y]').astype(int) + 1970
    g = np.asarray(gross_ret, dtype=float)
    c = np.asarray(cost, dtype=float)
    m = np.asarray(market, dtype=float)
    net = g - c
    ty = None
    if trade_dates is not None:
        ty = np.asarray(trade_dates, dtype='datetime64[D]').astype('datetime64[Y]').astype(int) + 1970
        tt = np.asarray(trade_turnover, dtype=float)

    def sharpe(v):
        sd = v.std(ddof=1)
        return float(v.mean() / sd * np.sqrt(TRADING_DAYS)) if len(v) > 2 and sd > 0 else None

    rows = []
    for yr in np.unique(years):
        s = years == yr
        ok = s & np.isfinite(m)
        beta = float(np.cov(net[ok], m[ok])[0, 1] / np.var(m[ok], ddof=1)) if ok.sum() > 20 and np.var(m[ok]) > 0 else None
        rows.append({
            'year': int(yr), 'n_days': int(s.sum()),
            'net_return': float(np.prod(1.0 + net[s]) - 1.0),
            'ann_vol': float(net[s].std(ddof=1) * np.sqrt(TRADING_DAYS)) if s.sum() > 2 else None,
            'sharpe_net': sharpe(net[s]), 'sharpe_gross': sharpe(g[s]),
            'beta': beta,
            'cost_bps': float(c[s].sum() * 1e4),
            'turnover': float(tt[ty == yr].sum()) if ty is not None else None,
        })
    return rows
