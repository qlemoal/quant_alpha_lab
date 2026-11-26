'''
GOAL: turn CRSP daily stock data (WRDS, new "CIZ" format, table crsp.dsf_v2) into the per-ticker CSV files that
    stage 02 reads. CRSP keeps the dead companies, so this is the survivorship-free source.

WHY CRSP
    - It contains every US common stock that ever traded on NYSE, AMEX or Nasdaq, DEAD ONES INCLUDED, from 1925.
    - Its returns are split-aware and include dividends, and the delisting return is part of the last return.
    - EPFL gives access through WRDS (account with an @epfl.ch address). Check on the library page. Terms of use
      forbid sharing the data: keep data/ out of git (it is in .gitignore).

THE LEGACY FORMAT IS OVER
    CRSP stopped the old flat-file format ("SIZ", tables crsp.dsf, crsp.dsenames, crsp.dsedelist) with the
    December 2024 data. For 2025 and 2026 you need the new tables. This module uses the new ones:
        crsp.dsf_v2               daily data. Columns used: permno, dlycaldt, dlyprc, dlyret, dlyretx, dlyvol,
                                  dlyopen, dlyhigh, dlylow, dlycap
        crsp.stksecurityinfohist  security history. Columns used: permno, ticker, primaryexch, secinfostartdt, secinfoenddt
                                  (also the filters for common stocks on NYSE, AMEX, Nasdaq)
    I wrote the names from memory of public WRDS and community documentation (the CRSP guide pages did not open
    from my environment), and I could NOT test them against a live
    WRDS account. `01_import_crsp.py --probe` prints the real column names. If one differs, the importer stops with
    a message that lists what it found. Change the COLUMNS dict below, nothing else.

HOW THE PRICES ARE BUILT (no reliance on CRSP's split factors)
    CRSP returns are exact for splits and dividends. So we rebuild both price paths from the returns:

        dlyretx  = return without dividends  --> Close      (split-adjusted price, the Yahoo "Close")
        dlyret   = total return              --> Adj Close  (splits and dividends, the Yahoo "Adj Close")

    Both paths are anchored on the last trading day of the stock (alive stock) or on the day BEFORE the last
    day (dead stock), at the real price |dlyprc| (CRSP shows a negative price when it is a bid/ask midpoint).
    Anchoring a dead stock one day earlier keeps Adj Close <= Close in the past, as the ETL check expects, and puts the
    delisting return in the last row of Adj Close only.
        Open, High, Low  = raw value x (Close / raw price), so they live on the same split basis as Close
        Volume           = raw volume x raw price / Close   (so Close x Volume = dollars traded, always)

THE DELISTING RETURN
    I BELIEVE that in the new format the delisting return is already inside dlyret of the last day. I could not open
    CRSP's own guide to confirm it. THE AUDIT IS THE SAFETY NET: if the delisting return is not there, the dead stocks
    do not crash on their last day, and 02b prints a WARN (share of crashes below 20%). If you see it, tell me.
    If it is MISSING for a stock that stopped trading
    (a delisting for cause with no known value), we assume -30% (Shumway 1997: about -30% for NYSE/AMEX performance
    delistings; Shumway and Warther 1999 give about -55% for Nasdaq). It is an ASSUMPTION, counted in the report.
    Dropping the stock instead would be the survivorship bias we want to remove.

NASDAQ VOLUME BEFORE 2004
    Nasdaq volume counted dealer-to-dealer trades twice. Dollar volume (our illiquidity signal and our universe
    rule) is then too high for Nasdaq stocks. Gao and Ritter (2010) give divisors: 2.0 before 2001-02-01, 1.8 before
    2002-01-01, 1.6 before 2004-01-01, 1.0 after. Applied by default (adjust_nasdaq_volume=True).
'''

import re

import numpy as np
import pandas as pd

YAHOO_ORDER = ['Date', 'Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']

COLUMNS = {
    'daily': {'permno': 'permno', 'date': 'dlycaldt', 'prc': 'dlyprc', 'ret': 'dlyret', 'retx': 'dlyretx',
              'vol': 'dlyvol', 'open': 'dlyopen', 'high': 'dlyhigh', 'low': 'dlylow'},
    'info': {'permno': 'permno', 'ticker': 'ticker', 'exch': 'primaryexch', 'start': 'secinfostartdt', 'end': 'secinfoenddt'},
}
CAP_COLUMN = 'dlycap'                      # market capitalisation in thousands of dollars, used only to choose which permnos to download
TERMINAL_RETURN = -0.30                    # assumed delisting return when CRSP has none (see header)
NASDAQ_DIVISORS = [('2001-02-01', 2.0), ('2002-01-01', 1.8), ('2004-01-01', 1.6)]

SQL_PERMNOS = '''
    SELECT a.permno
    FROM crsp.dsf_v2 a
    JOIN crsp.stksecurityinfohist b
      ON a.permno = b.permno AND a.dlycaldt BETWEEN b.secinfostartdt AND b.secinfoenddt
    WHERE a.dlycaldt BETWEEN '{start}' AND '{end}'
      AND b.sharetype = 'NS' AND b.securitytype = 'EQTY' AND b.securitysubtype = 'COM'
      AND b.usincflg = 'Y' AND b.issuertype IN ('ACOR', 'CORP')
      AND b.primaryexch IN ('N', 'A', 'Q')
    GROUP BY a.permno
    HAVING max(a.{cap}) >= {min_cap}
    ORDER BY a.permno
'''
SQL_DAILY = '''
    SELECT permno, dlycaldt, dlyprc, dlyret, dlyretx, dlyvol, dlyopen, dlyhigh, dlylow
    FROM crsp.dsf_v2
    WHERE permno IN ({permnos}) AND dlycaldt BETWEEN '{start}' AND '{end}'
    ORDER BY permno, dlycaldt
'''
SQL_INFO = '''
    SELECT permno, ticker, primaryexch, secinfostartdt, secinfoenddt
    FROM crsp.stksecurityinfohist
    WHERE permno IN ({permnos})
    ORDER BY permno, secinfostartdt
'''


def check_columns(df: pd.DataFrame, mapping: dict, table: str) -> None:
    '''Stop with a clear message if the table lacks a column the importer needs.'''
    missing = [v for v in mapping.values() if v not in df.columns]
    if missing:
        raise ValueError(f'{table} lacks the columns {missing}. It has {sorted(df.columns)}. '
                         f'Edit COLUMNS in src/data/crsp.py (run 01_import_crsp.py --probe to see the real names).')


def nasdaq_divisor(dates: pd.Series) -> np.ndarray:
    '''Gao-Ritter divisor of the Nasdaq volume for each date (1.0 from 2004).'''
    d = pd.to_datetime(dates)
    out = np.ones(len(d))
    for cutoff, div in reversed(NASDAQ_DIVISORS):          # latest cutoff first, earlier overrides
        out[d < pd.Timestamp(cutoff)] = div
    return out


def safe_name(ticker: str, permno) -> str:
    '''File-safe, unique id: tickers are reused by CRSP over time, the permno never is. AAPL -> AAPL_14593.'''
    t = re.sub(r'[^A-Za-z0-9.\-]', '', str(ticker)) or 'NA'
    return f'{t}_{int(permno)}'


def convert_one(g: pd.DataFrame, is_dead: bool, cols: dict = COLUMNS['daily'], adjust_nasdaq: bool = True,
                exch: "pd.Series | None" = None, terminal_return: float = TERMINAL_RETURN) -> "tuple[pd.DataFrame, dict]":
    '''
    One permno (rows sorted by date) -> Yahoo-layout table + a small report. See the module header for the method.
    g: the daily rows. is_dead: the stock stops trading before the end of the data.
    exch: optional Series (same index as g) with the primary exchange letter, for the Nasdaq volume adjustment.
    '''
    c = cols
    g = g.sort_values(c['date']).reset_index(drop=True)
    px = g[c['prc']].abs()
    g = g[px.notna() & (px > 0)].reset_index(drop=True)
    rep = {'terminal_assumed': 0, 'ret_missing': 0, 'ohl_filled': 0}
    if len(g) < 2:
        return pd.DataFrame(columns=YAHOO_ORDER), {**rep, 'skipped': 1}
    px = g[c['prc']].abs().to_numpy(float)
    ret = g[c['ret']].to_numpy(float, copy=True)          # copies: we write into them below
    retx = g[c['retx']].to_numpy(float, copy=True)

    if is_dead and np.isnan(ret[-1]):
        ret[-1] = terminal_return
        retx[-1] = terminal_return if np.isnan(retx[-1]) else retx[-1]
        rep['terminal_assumed'] = 1
    rep['ret_missing'] = int(np.isnan(ret[1:]).sum())
    ret = np.where(np.isnan(ret), 0.0, ret)
    retx = np.where(np.isnan(retx), ret, retx)             # no price-only return: use the total return
    ret[0] = 0.0
    retx[0] = 0.0                                          # the first row has no previous price

    P = np.cumprod(1.0 + retx)
    A = np.cumprod(1.0 + ret)
    n = len(g)
    anchor = n - 2 if is_dead else n - 1                   # dead: the day before the last (see header)
    close = px[anchor] * P / P[anchor]
    adj = px[anchor] * A / A[anchor]

    f = close / px                                         # raw price -> split-adjusted price
    if is_dead:
        f[-1] = f[-2]                                      # the last row's Close is not a trade price
    def ohl(name):
        raw = g[c[name]].abs().to_numpy(float)
        return np.where(np.isfinite(raw) & (raw > 0), raw * f, np.nan)
    o, h, l = ohl('open'), ohl('high'), ohl('low')
    rep['ohl_filled'] = int(np.isnan(o).sum())
    o = np.where(np.isnan(o), close, o)
    h = np.where(np.isnan(h), np.maximum(close, o), h)
    l = np.where(np.isnan(l), np.minimum(close, o), l)
    h = np.maximum.reduce([h, o, close])                   # an adjusted open can sit above the raw high by rounding
    l = np.minimum.reduce([l, o, close])

    vol = g[c['vol']].to_numpy(float)
    if adjust_nasdaq and exch is not None:
        e = exch.reindex(g.index) if len(exch) != n else exch.reset_index(drop=True)
        div = np.where(e.to_numpy() == 'Q', nasdaq_divisor(g[c['date']]), 1.0)
        vol = vol / div
    vol = np.nan_to_num(vol / f, nan=0.0)                  # Close x Volume = dollars traded (before the Nasdaq divisor)

    out = pd.DataFrame({
        'Date': pd.to_datetime(g[c['date']]).dt.strftime('%Y-%m-%d'),
        'Adj Close': adj, 'Close': close, 'High': h, 'Low': l, 'Open': o,
        'Volume': np.round(vol).astype('int64'),
    })[YAHOO_ORDER]
    return out, rep


def convert_daily(daily: pd.DataFrame, info: pd.DataFrame, end_date, dead_tolerance_days: int = 10,
                  adjust_nasdaq: bool = True, terminal_return: float = TERMINAL_RETURN) -> "tuple[dict, pd.DataFrame, dict]":
    '''
    daily: rows of crsp.dsf_v2 (any number of permnos). info: rows of crsp.stksecurityinfohist for the same permnos.
    end_date: the last date of the download (a stock whose last row is more than dead_tolerance_days before it is DEAD).
    Returns (frames {id: Yahoo table}, security_master table, report).
    security_master: id, permno, ticker, first_date, last_date, is_dead, terminal_assumed.
    '''
    cd, ci = COLUMNS['daily'], COLUMNS['info']
    check_columns(daily, cd, 'crsp.dsf_v2')
    check_columns(info, ci, 'crsp.stksecurityinfohist')
    d = daily.copy()
    d[cd['date']] = pd.to_datetime(d[cd['date']])
    inf = info.copy()
    inf[ci['start']] = pd.to_datetime(inf[ci['start']])
    inf[ci['end']] = pd.to_datetime(inf[ci['end']])

    # exchange of each row (as of its date) for the Nasdaq volume adjustment
    d = d.sort_values(cd['date'])
    m = pd.merge_asof(d, inf.sort_values(ci['start'])[[ci['permno'], ci['start'], ci['exch']]],
                      left_on=cd['date'], right_on=ci['start'], by=ci['permno'], direction='backward')
    exch_all = m[ci['exch']].set_axis(d.index)

    frames, master = {}, []
    rep = {'n_permnos': 0, 'n_dead': 0, 'terminal_assumed': 0, 'ret_missing': 0, 'ohl_filled': 0, 'skipped': 0}
    end_ts = pd.Timestamp(end_date)
    for permno, g in d.groupby(cd['permno']):
        last = g[cd['date']].max()
        dead = (end_ts - last).days > dead_tolerance_days
        out, r = convert_one(g, dead, cd, adjust_nasdaq, exch_all.loc[g.index], terminal_return)
        rep['n_permnos'] += 1
        rep['skipped'] += r.get('skipped', 0)
        if out.empty:
            continue
        i = inf[inf[ci['permno']] == permno].sort_values(ci['start'])
        tk = i[ci['ticker']].iloc[-1] if len(i) else str(permno)          # the ticker the stock had at the end
        name = safe_name(tk, permno)
        frames[name] = out
        rep['n_dead'] += int(dead)
        for k in ('terminal_assumed', 'ret_missing', 'ohl_filled'):
            rep[k] += r.get(k, 0)
        master.append({'id': name, 'permno': int(permno), 'ticker': tk, 'first_date': out['Date'].iloc[0],
                       'last_date': out['Date'].iloc[-1], 'is_dead': bool(dead), 'terminal_assumed': r.get('terminal_assumed', 0)})
    return frames, pd.DataFrame(master), rep
