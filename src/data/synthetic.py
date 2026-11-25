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
    No stock is ever delisted: the synthetic market HAS NO survivorship bias on purpose (stage 02 to 09 then show
    what the pipeline does on clean data).

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

    # ---- prices
    dates = pd.bdate_range(cfg.start, periods=T)
    day_idx = np.arange(T)[:, None]
    ex_day = ((day_idx - div_phase[None, :]) % 63 == 0) & (day_idx > 0)
    price_ret = ret - ex_day * (div_yield[None, :] / 4.0)                    # price drops on the ex-day
    close = start_price[None, :] * np.exp(np.cumsum(price_ret, axis=0))
    total = start_price[None, :] * np.exp(np.cumsum(ret, axis=0))
    adj_close = total * (close[-1] / total[-1])[None, :]                     # anchored on the last day, as Yahoo does

    prev_close = np.vstack([start_price[None, :], close[:-1]])
    open_ = prev_close * np.exp(0.35 * price_ret + 0.004 * gap_noise)
    high = np.maximum(open_, close) * (1 + 0.004 * range_hi)
    low = np.minimum(open_, close) / (1 + 0.004 * range_lo)
    dollar_vol = 2e7 * np.exp(0.9 * z_liq)[None, :] * np.exp(0.25 * vol_noise + 0.5 * np.abs(ret) / sigma[None, :] - 0.5)
    volume = np.round(dollar_vol / close)

    return {
        'dates': dates, 'tickers': np.array([f'S{i:03d}' for i in range(N)]),
        'industry': industry, 'beta': beta, 'z_vol': z_vol, 'z_liq': z_liq,
        'total_ret': ret, 'price_ret': price_ret,
        'close': close, 'adj_close': adj_close, 'open': open_, 'high': high, 'low': low, 'volume': volume,
    }


def ticker_frame(mkt: dict, j: int) -> pd.DataFrame:
    '''One stock as a yfinance-style table (rounded to 6 decimals).'''
    return pd.DataFrame({
        'Date': mkt['dates'].strftime('%Y-%m-%d'),
        'Adj Close': mkt['adj_close'][:, j].round(6),
        'Close': mkt['close'][:, j].round(6),
        'High': mkt['high'][:, j].round(6),
        'Low': mkt['low'][:, j].round(6),
        'Open': mkt['open'][:, j].round(6),
        'Volume': mkt['volume'][:, j].astype('int64'),
    })[YAHOO_ORDER]


def market_digest(mkt: dict) -> str:
    '''SHA-256 of the rounded price and volume arrays. Same seed and NumPy: same digest.'''
    h = hashlib.sha256()
    for key in ('adj_close', 'close', 'high', 'low', 'open'):
        h.update(np.ascontiguousarray(mkt[key].round(6)).tobytes())
    h.update(np.ascontiguousarray(mkt['volume'].astype('int64')).tobytes())
    return h.hexdigest()


def write_market(mkt: dict, cfg: SyntheticConfig, out_dir: Path) -> dict:
    '''
    Write one CSV per ticker, plus config.json (parameters, digest) and truth.json (the planted structure).
    Returns the config dict that was written.
    '''
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for j, t in enumerate(mkt['tickers']):
        ticker_frame(mkt, j).to_csv(out_dir / f'{t}.csv', index=False)
    info = {'generator': 'src/data/synthetic.py', 'config': asdict(cfg), 'digest': market_digest(mkt),
            'n_tickers': int(len(mkt['tickers'])), 'n_days': int(len(mkt['dates']))}
    (out_dir / 'config.json').write_text(json.dumps(info, indent=2))
    truth = {
        'planted_effects': {'low_volatility': -cfg.c_vol, 'illiquidity': -cfg.c_liq,
                            'reversal': -cfg.c_rev, 'momentum': cfg.c_mom},
        'industry': {str(t): int(k) for t, k in zip(mkt['tickers'], mkt['industry'])},
    }
    (out_dir / 'truth.json').write_text(json.dumps(truth, indent=2))
    return info
