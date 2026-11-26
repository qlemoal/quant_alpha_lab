'''
GOAL: build a fake stock market in which WE decide what is predictable.
    The pipeline is then tested against a known truth, and two machines can compare numbers.

WHY
    On real data we never know the right answer. Here we do:
      - the reversal, momentum, low-volatility and illiquidity effects are planted with known strength;
      - the industries are known, so we can check the clusters;
      - nothing else is predictable, so any other signal that passes the gates is a bug.
    The same seed gives the same market on every machine (NumPy PCG64, no Polars hash).
    WHAT IT CANNOT DO: say anything about REAL alpha. The strengths are my assumptions.

THE MODEL (daily log return of stock i on day t)

    r_it = beta_i * m_t                      market factor           m_t   ~ N(mu, 0.95%)
         + gamma_i * f_{k(i),t}              industry factor         f     ~ N(0, 0.6%), k(i) = industry of i
         + sigma_i * e_it                    idiosyncratic noise     e     ~ Student t (5 dof), unit variance
         + drift_it                          the PLANTED effects

    drift_it = - c_vol  * z_vol_i            low volatility:  stocks with a high noise level earn less   (fixed attribute)
               - c_liq  * z_liq_i            illiquidity:     stocks with a high volume earn less        (fixed attribute)
               - c_rev  * Z(past 5-day return)       short-term reversal   (cross-sectional z-score, recomputed daily)
               + c_mom  * Z(return t-252 .. t-21)    momentum              (skips the last month)

    sigma_i = 1.2% * exp(0.35 * z_vol_i)       so the volatility FEATURE (vol20, vol60 ...) measures z_vol_i with noise
    dollar volume_i = 2e7 * exp(0.9 * z_liq_i) so the log_adv features measure z_liq_i

THE PRICE FILES
    Same layout as yfinance with auto_adjust=False: Date, Adj Close, Close, High, Low, Open, Volume.
    Close is adjusted for splits only (there are no splits here), Adj Close also for dividends.
    Each stock pays a dividend every 63 days. The ex-day price drops by the dividend, the total return does not.
    By default no stock is delisted: no dead stocks, no survivorship issue (stage 02 to 09 show what the pipeline
    does on clean data). With delist_rate > 0 stocks DIE, and the risk is higher after a bad year:
        hazard(i,t) = delist_rate / 252 x exp(-delist_sensitivity x z_it),  z_it = cross-sectional z-score of the past-year return
        the delisting return is a crash (-30% to -99%) with probability 1 / (1 + exp(1.5 z + 0.5)), else a take-over (0% to +35%)
    The delisting return is added to the ADJUSTED close of the last row only. The Close is the last trade.
    write_market(..., drop_dead=True) writes only the survivors: the file that a free source gives you.
    The difference between the two files is the survivorship bias, measured on a market where the truth is known.

REPRODUCIBILITY
    market_digest(frame) = SHA-256 of the numbers rounded to 6 decimals. Two machines with the same NumPy
    should give the same digest. If not, compare the numbers with a tolerance instead.
'''

import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

YAHOO_ORDER = ['Date', 'Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']


@dataclass(frozen=True)
class SyntheticConfig:
    n_tickers: int = 300
    n_days: int = 2520                 # about 10 years
    start: str = '2016-01-04'
    n_industries: int = 8
    seed: int = 12345
    # market and industry factors (daily)
    market_mu: float = 0.0003
    market_vol: float = 0.0095
    industry_vol: float = 0.006
    base_idio_vol: float = 0.012
    # planted effects: drift in daily log return per one standard deviation of the attribute
    c_vol: float = 0.00012             # low volatility
    c_liq: float = 0.00008             # illiquidity
    c_rev: float = 0.0008              # short-term reversal
    c_mom: float = 0.00009             # momentum
    # dividends: annual yield between 0 and this, paid every 63 days
    max_dividend_yield: float = 0.02
    # delistings (0 = none, the default: the market has no dead stocks and the reference run does not change)
    delist_rate: float = 0.0           # base yearly probability that a stock is delisted
    delist_sensitivity: float = 1.0    # how much a bad past year raises the risk: hazard x exp(-sensitivity x z of the past-year return)


def _zscore_rows(x: np.ndarray) -> np.ndarray:
    '''Cross-sectional z-score of each row (one row = one day). A constant row gives zeros.'''
    sd = x.std(axis=1, keepdims=True)
    sd = np.where(sd > 0, sd, 1.0)
    return (x - x.mean(axis=1, keepdims=True)) / sd


def simulate_market(cfg: SyntheticConfig = SyntheticConfig()) -> dict:
    '''
    Simulate the market. Returns a dict of arrays:
        dates (n_days,), tickers (n,), industry (n,), beta (n,), z_vol (n,), z_liq (n,),
        total_ret, price_ret (n_days, n), close, adj_close, open, high, low, volume (n_days, n)
    All randomness comes from one PCG64 generator seeded with cfg.seed.
    '''
    rng = np.random.default_rng(cfg.seed)
    T, N, K = cfg.n_days, cfg.n_tickers, cfg.n_industries

    # ---- fixed attributes of the stocks
    industry = rng.integers(0, K, size=N)
    beta = rng.uniform(0.6, 1.4, size=N)
    gamma = rng.uniform(0.7, 1.3, size=N)
    z_vol = rng.standard_normal(N)
    z_liq = rng.standard_normal(N)
    sigma = cfg.base_idio_vol * np.exp(0.35 * z_vol)
    div_yield = rng.uniform(0.0, cfg.max_dividend_yield, size=N)
    div_phase = rng.integers(0, 63, size=N)
    start_price = 50.0 * np.exp(0.5 * rng.standard_normal(N))

    # ---- shocks (drawn first, so that the numbers do not depend on the loop)
    market = cfg.market_mu + cfg.market_vol * rng.standard_normal(T)
    ind_f = cfg.industry_vol * rng.standard_normal((T, K))
    dof = 5
    eps = rng.standard_t(dof, size=(T, N)) / np.sqrt(dof / (dof - 2))        # unit variance, fat tails
    gap_noise = rng.standard_normal((T, N))
    range_hi = np.abs(rng.standard_normal((T, N)))
    range_lo = np.abs(rng.standard_normal((T, N)))
    vol_noise = rng.standard_normal((T, N))

    base = beta[None, :] * market[:, None] + gamma[None, :] * ind_f[:, industry] + sigma[None, :] * eps
    static_drift = -cfg.c_vol * z_vol - cfg.c_liq * z_liq                    # (N,)

    # ---- day loop: reversal and momentum depend on the stock's own past returns
    ret = np.zeros((T, N))
    csum = np.zeros((T + 1, N))                                              # csum[t] = sum of ret[:t]
    for t in range(T):
        drift = static_drift.copy()
        if t >= 6:
            past5 = csum[t] - csum[t - 5]
            drift += -cfg.c_rev * _zscore_rows(past5[None, :])[0]
        if t >= 253:
            mom = csum[t - 21] - csum[t - 252]
            drift += cfg.c_mom * _zscore_rows(mom[None, :])[0]
        ret[t] = base[t] + drift
        csum[t + 1] = csum[t] + ret[t]

    # ---- delistings (own generator: every draw above is unchanged, so delist_rate = 0 gives the same market as before)
    last_idx = np.full(N, T - 1)
    dlret = np.zeros(N)
    if cfg.delist_rate > 0:
        rng_d = np.random.default_rng([cfg.seed, 1])
        trail = np.zeros((T, N))
        trail[252:] = csum[252:T] - csum[:T - 252]                            # return of the year before day t (known at t)
        z = np.vstack([_zscore_rows(trail[t][None, :]) if t >= 252 else np.zeros((1, N)) for t in range(T)])
        hazard = np.minimum(cfg.delist_rate / 252.0 * np.exp(-cfg.delist_sensitivity * z), 0.05)
        event = rng_d.random((T, N)) < hazard
        event[:126] = False                                                    # no stock dies in its first half year
        has = event.any(axis=0)
        first = event.argmax(axis=0)
        last_idx = np.where(has, first, T - 1)
        z_at = z[first, np.arange(N)]
        p_bad = 1.0 / (1.0 + np.exp(1.5 * z_at + 0.5))
        bad = rng_d.random(N) < p_bad
        dl = np.where(bad, -rng_d.uniform(0.30, 0.99, N), rng_d.uniform(0.0, 0.35, N))
        dlret = np.where(has, dl, 0.0)
    dead = last_idx < T - 1

    # ---- prices
    dates = pd.bdate_range(cfg.start, periods=T)
    day_idx = np.arange(T)[:, None]
    ex_day = ((day_idx - div_phase[None, :]) % 63 == 0) & (day_idx > 0)
    price_ret = ret - ex_day * (div_yield[None, :] / 4.0)                    # price drops on the ex-day
    close = start_price[None, :] * np.exp(np.cumsum(price_ret, axis=0))
    total = start_price[None, :] * np.exp(np.cumsum(ret, axis=0))
    cols = np.arange(N)
    adj_close = total * (close[last_idx, cols] / total[last_idx, cols])[None, :]   # anchored on the last TRADING day, as Yahoo does
    adj_close[last_idx, cols] *= (1.0 + dlret)                               # the delisting return lives in the adjusted close only

    prev_close = np.vstack([start_price[None, :], close[:-1]])
    open_ = prev_close * np.exp(0.35 * price_ret + 0.004 * gap_noise)
    high = np.maximum(open_, close) * (1 + 0.004 * range_hi)
    low = np.minimum(open_, close) / (1 + 0.004 * range_lo)
    dollar_vol = 2e7 * np.exp(0.9 * z_liq)[None, :] * np.exp(0.25 * vol_noise + 0.5 * np.abs(ret) / sigma[None, :] - 0.5)
    volume = np.round(dollar_vol / close)

    # nothing exists after the last trading day of a dead stock
    after = np.arange(T)[:, None] > last_idx[None, :]
    for arr in (close, adj_close, open_, high, low, volume):
        arr[after] = np.nan

    return {
        'dates': dates, 'tickers': np.array([f'S{i:03d}' for i in range(N)]),
        'industry': industry, 'beta': beta, 'z_vol': z_vol, 'z_liq': z_liq,
        'total_ret': ret, 'price_ret': price_ret, 'last_idx': last_idx, 'dlret': dlret, 'dead': dead,
        'close': close, 'adj_close': adj_close, 'open': open_, 'high': high, 'low': low, 'volume': volume,
    }


def ticker_frame(mkt: dict, j: int) -> pd.DataFrame:
    '''One stock as a yfinance-style table (rounded to 6 decimals).'''
    n = int(mkt['last_idx'][j]) + 1                                          # rows up to the last trading day
    return pd.DataFrame({
        'Date': mkt['dates'][:n].strftime('%Y-%m-%d'),
        'Adj Close': mkt['adj_close'][:n, j].round(6),
        'Close': mkt['close'][:n, j].round(6),
        'High': mkt['high'][:n, j].round(6),
        'Low': mkt['low'][:n, j].round(6),
        'Open': mkt['open'][:n, j].round(6),
        'Volume': mkt['volume'][:n, j].astype('int64'),
    })[YAHOO_ORDER]


def market_digest(mkt: dict) -> str:
    '''SHA-256 of the rounded price and volume arrays. Same seed and NumPy: same digest.'''
    h = hashlib.sha256()
    for key in ('adj_close', 'close', 'high', 'low', 'open'):
        h.update(np.ascontiguousarray(np.nan_to_num(mkt[key].round(6), nan=0.0)).tobytes())
    h.update(np.ascontiguousarray(np.nan_to_num(mkt['volume'], nan=0.0).astype('int64')).tobytes())
    return h.hexdigest()


def write_market(mkt: dict, cfg: SyntheticConfig, out_dir: Path, drop_dead: bool = False) -> dict:
    '''
    Write one CSV per ticker, plus config.json (parameters, digest) and truth.json (the planted structure).
    drop_dead=True writes only the stocks that never delisted: the "survivors only" file of a free source.
    Returns the config dict that was written.
    '''
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_written = 0
    for j, t in enumerate(mkt['tickers']):
        if drop_dead and mkt['dead'][j]:
            continue
        ticker_frame(mkt, j).to_csv(out_dir / f'{t}.csv', index=False)
        n_written += 1
    info = {'generator': 'src/data/synthetic.py', 'config': asdict(cfg), 'digest': market_digest(mkt),
            'n_tickers': int(n_written), 'n_days': int(len(mkt['dates'])), 'n_dead_in_market': int(mkt['dead'].sum()),
            'drop_dead': bool(drop_dead)}
    (out_dir / 'config.json').write_text(json.dumps(info, indent=2))
    truth = {
        'planted_effects': {'low_volatility': -cfg.c_vol, 'illiquidity': -cfg.c_liq,
                            'reversal': -cfg.c_rev, 'momentum': cfg.c_mom},
        'industry': {str(t): int(k) for t, k in zip(mkt['tickers'], mkt['industry'])},
        'dead': {str(t): {'last_day': str(mkt['dates'][int(l)].date()), 'delisting_return': float(r)}
                 for t, l, r, d in zip(mkt['tickers'], mkt['last_idx'], mkt['dlret'], mkt['dead']) if d},
    }
    (out_dir / 'truth.json').write_text(json.dumps(truth, indent=2))
    return info
