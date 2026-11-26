'''
GOAL: a universe that uses ONLY what was known on each date ("point in time"), and the checks that prove it.
    This is the part of the project that removes SURVIVORSHIP BIAS. Read this header first.

THE PROBLEM, IN ONE PICTURE
    A list of today's S&P 500 members, back-filled to the year 2000, contains only companies that SURVIVED
    and GREW until today. Bankrupt, delisted and demoted companies are absent. Two things go wrong:

        (1) the PRICES of the dead companies are missing      -> their losses are never traded
        (2) the UNIVERSE of each date is chosen with hindsight -> "stocks that will be in the index" are winners

    No script can invent (1). The prices of dead companies must come from a vendor that keeps them
    (CRSP, Sharadar, Norgate, ...). See docs/survivorship.md. This file solves (2) and the bookkeeping of (1).

THE FOUR PARTS

    [A] rule_universe          the universe is a RULE on past data: "the top N stocks by average dollar volume,
                               re-ranked every k days". Computed from prices only. No vendor list needed.
                               This is "as if we had it live": on each date it uses data up to that date.
    [B] membership_*           a vendor's index list (S&P 500 additions and removals) turned into a per-date mask.
    [C] dead_tickers           which tickers stop trading before the end of the data (delisted, merged, bankrupt).
    [D] audit_universe         numbers that tell you whether the data still look like "survivors only".

WHAT A POINT-IN-TIME UNIVERSE MEANS FOR EACH STAGE
    02b  writes universe.parquet [date, ticker, in_universe]
    04   keeps only the rows with in_universe = True BEFORE it normalises the signals across stocks,
         so a signal is ranked among the stocks that were investable that day, nothing else.
    04   labels: a stock that delists keeps its LAST price as the exit price (the delisting return is
         already inside that price), so the crash of a dying stock is in the labels. Without this the
         last h+1 labels of every dead stock are null, and exactly the worst days vanish.

WHAT THIS FILE CANNOT DO
    - It cannot repair a price file that lacks the dead companies. audit_universe tells you if that is the case.
    - Dollar volume is the only size measure available from prices alone. Market cap needs shares outstanding.
'''

import numpy as np
import polars as pl

DEFAULT_RULE = {
    'top_n': 1000,               # keep the N stocks with the highest average dollar volume
    'adv_window': 60,            # days in the average dollar volume
    'min_adv': 1_000_000.0,      # and at least this many dollars per day
    'min_history': 126,          # rows of history needed before a stock can enter
    'reconstitute_every': 21,    # re-rank every k trading days (about monthly); in between the set only loses dead names
}


# =============================================================================
# [C] DEAD TICKERS
# =============================================================================

def dead_tickers(ohlcv: pl.DataFrame, tolerance_days: int = 10) -> pl.DataFrame:
    '''
    One row per ticker: first_date, last_date, n_rows, is_dead.
    A ticker is DEAD if its last date is more than `tolerance_days` calendar days before the last date of the
    whole table. (It stopped trading: delisted, merged, bankrupt. Or the vendor stopped the file: same effect.)
    '''
    last_all = ohlcv['date'].max()
    out = (
        ohlcv.group_by('ticker')
        .agg(pl.col('date').min().alias('first_date'), pl.col('date').max().alias('last_date'), pl.len().alias('n_rows'))
        .with_columns(((pl.lit(last_all) - pl.col('last_date')).dt.total_days() > tolerance_days).alias('is_dead'))
        .sort('ticker')
    )
    return out


# =============================================================================
# [A] RULE UNIVERSE (computed from prices, point in time)
# =============================================================================

def rule_universe(ohlcv: pl.DataFrame, top_n: int = DEFAULT_RULE['top_n'], adv_window: int = DEFAULT_RULE['adv_window'],
                  min_adv: float = DEFAULT_RULE['min_adv'], min_history: int = DEFAULT_RULE['min_history'],
                  reconstitute_every: int = DEFAULT_RULE['reconstitute_every']) -> pl.DataFrame:
    '''
    The universe as a rule on PAST data. Returns [date, ticker, in_universe, adv, rank].

        adv     = mean over the last `adv_window` rows of close_raw * volume (split-adjusted price x split-adjusted
                  volume = dollars traded, the same on every share basis)
        every `reconstitute_every` trading days: eligible = adv >= min_adv and history >= min_history rows;
        the `top_n` eligible stocks by adv are IN. The set is kept until the next re-ranking.
        A stock that stops trading has no more rows, so it leaves by itself.

    No look-ahead: adv at date t uses rows up to and including t. The signals of date t are also built
    from the close of t, so the information set is the same.
    Why dollar volume and not a price filter: the price in the file is adjusted for splits to TODAY's share count,
    so "price >= 5" would be wrong for a stock that split many times. Dollar volume does not depend on the basis.
    The rank of a stock on a non-re-ranking date is the rank of its last re-ranking (null before its first one).
    '''
    need = {'ticker', 'date', 'close_raw', 'volume'}
    if not need <= set(ohlcv.columns):
        raise ValueError(f'ohlcv lacks the columns {sorted(need - set(ohlcv.columns))}')
    if top_n < 1 or reconstitute_every < 1 or adv_window < 1:
        raise ValueError('top_n, reconstitute_every and adv_window must be >= 1')

    df = ohlcv.select(['ticker', 'date', 'close_raw', 'volume']).sort(['ticker', 'date'])
    df = df.with_columns(
        (pl.col('close_raw') * pl.col('volume').cast(pl.Float64)).alias('dv'),
        pl.int_range(pl.len()).over('ticker').alias('_i'),
    ).with_columns(
        pl.col('dv').rolling_mean(window_size=adv_window, min_samples=adv_window).over('ticker').alias('adv'),
    )

    dates = df['date'].unique().sort()
    recon_dates = dates.gather(list(range(0, len(dates), reconstitute_every)))
    df = df.with_columns(pl.col('date').is_in(recon_dates.implode()).alias('_recon'))

    eligible = (pl.col('adv') >= min_adv) & (pl.col('_i') + 1 >= min_history)
    df = df.with_columns(pl.when(eligible).then(pl.col('adv')).otherwise(None).alias('_adv_elig'))
    df = df.with_columns(pl.col('_adv_elig').rank('ordinal', descending=True).over('date').alias('_rank'))
    df = df.with_columns(
        pl.when(pl.col('_recon')).then(pl.col('_rank')).otherwise(None).alias('_rank_recon'),
    ).sort(['ticker', 'date'])
    # the rank of the last re-ranking is kept until the next one (the stock is a member while rank <= top_n)
    df = df.with_columns(
        pl.when(pl.col('_recon')).then(pl.col('_rank')).otherwise(None).alias('rank')
    ).with_columns(pl.col('rank').forward_fill().over('ticker'))
    df = df.with_columns((pl.col('rank').is_not_null() & (pl.col('rank') <= top_n)).alias('in_universe'))
    return df.select(['date', 'ticker', 'in_universe', 'adv', 'rank'])


# =============================================================================
# [B] VENDOR MEMBERSHIP (index lists)
# =============================================================================

def membership_from_events(events: pl.DataFrame, first_date=None) -> pl.DataFrame:
    '''
    Turn an event list into intervals. events: [date, ticker, action] with action 'added' or 'removed'
    (case-insensitive; other words are ignored). Returns [ticker, start, end] with end = null while still a member.

    A 'removed' event with no earlier 'added' means "member since the start of the data": start = first_date
    (default: the earliest event date). Two 'added' in a row open nothing new. Events are used in date order.
    '''
    ev = (events.select(['date', 'ticker', pl.col('action').cast(pl.Utf8).str.to_lowercase().alias('action')])
          .filter(pl.col('action').is_in(['added', 'removed'])).sort(['ticker', 'date']))
    if ev.height == 0:
        return pl.DataFrame(schema={'ticker': pl.Utf8, 'start': pl.Date, 'end': pl.Date})
    t0 = first_date if first_date is not None else ev['date'].min()
    rows = []
    for tk, g in ev.group_by('ticker', maintain_order=True):
        open_start = None
        for d, a in zip(g['date'].to_list(), g['action'].to_list()):
            if a == 'added':
                if open_start is None:
                    open_start = d
            else:
                rows.append((tk[0], open_start if open_start is not None else t0, d))
                open_start = None
        if open_start is not None:
            rows.append((tk[0], open_start, None))
    return pl.DataFrame(rows, schema={'ticker': pl.Utf8, 'start': pl.Date, 'end': pl.Date}, orient='row')


def validate_membership(m: pl.DataFrame) -> None:
    '''Raise on a membership table that cannot be right: missing columns, end before start, overlapping intervals.'''
    if not {'ticker', 'start', 'end'} <= set(m.columns):
        raise ValueError(f'membership needs the columns ticker, start, end. Got {m.columns}')
    bad = m.filter(pl.col('end').is_not_null() & (pl.col('end') < pl.col('start')))
    if bad.height:
        raise ValueError(f'{bad.height} membership rows end before they start, e.g. {bad.head(2).to_dicts()}')
    srt = m.sort(['ticker', 'start']).with_columns(pl.col('end').shift(1).over('ticker').alias('_prev_end'))
    ov = srt.filter(pl.col('_prev_end').is_not_null() & (pl.col('start') <= pl.col('_prev_end')))
    if ov.height:
        raise ValueError(f'{ov.height} overlapping membership intervals for the same ticker, e.g. {ov.head(2).to_dicts()}')


def membership_mask(keys: pl.DataFrame, membership: pl.DataFrame) -> pl.DataFrame:
    '''
    keys: [date, ticker]. membership: [ticker, start, end]. Returns keys with in_universe = True when
    start <= date <= end (end null = still a member). The interval is closed on both sides.
    '''
    validate_membership(membership)
    # intervals of one ticker do not overlap (validated), so the latest interval that STARTED on or before the date
    # is the only candidate. As-of join, then check the end.
    k = keys.select(['date', 'ticker']).sort('date')
    m = membership.select(['ticker', 'start', 'end']).sort('start')
    j = k.join_asof(m, left_on='date', right_on='start', by='ticker', strategy='backward')
    out = j.select(['date', 'ticker',
                    (pl.col('start').is_not_null() & (pl.col('end').is_null() | (pl.col('date') <= pl.col('end')))).alias('in_universe')])
    return out.sort(['ticker', 'date'])


# =============================================================================
# [D] AUDIT
# =============================================================================

def audit_universe(ohlcv: pl.DataFrame, universe: "pl.DataFrame | None" = None, tolerance_days: int = 10) -> dict:
    '''
    Numbers that show whether the price file still looks like "survivors only".

    n_tickers, n_dead, share_dead      tickers that stop trading before the end of the data
    delist_rate_per_year               dead tickers by calendar year of their last date / tickers alive that year
    members_min / median / max         size of the universe per date (if a universe table is given)
    last_return_median, share_crash    the last-day return of the dead tickers: median, and the share below -20%
    verdict                            'FAIL', 'WARN' or 'OK', with the reasons in `notes`

    Rules (they are heuristics, written to catch the typical failure, a back-filled list of today's index):
      FAIL  no dead ticker at all in a file that covers more than 3 years
      WARN  fewer than 10% of the tickers are dead in a file that covers more than 10 years
      WARN  the median yearly delisting rate is below 1% (US all-cap markets lose several percent per year)
      WARN  fewer than 20% of the dead tickers have a last-day return below -20% AND there are more than 20 dead
            tickers (in real data a clear share of delistings are crashes; this also catches a vendor that
            drops the delisting return)
    '''
    d = dead_tickers(ohlcv, tolerance_days)
    n, n_dead = d.height, int(d['is_dead'].sum())
    first, last = ohlcv['date'].min(), ohlcv['date'].max()
    years_span = (last - first).days / 365.25
    notes = []

    # yearly delisting rate
    ds = d.with_columns(pl.col('first_date').dt.year().alias('fy'), pl.col('last_date').dt.year().alias('ly'))
    rates = {}
    for y in range(first.year, last.year):                       # the last year is incomplete by construction
        alive = ds.filter((pl.col('fy') <= y) & (pl.col('ly') >= y)).height
        died = ds.filter(pl.col('is_dead') & (pl.col('ly') == y)).height
        if alive:
            rates[y] = died / alive
    med_rate = float(np.median(list(rates.values()))) if rates else float('nan')

    # last-day return of dead tickers
    dead_names = d.filter(pl.col('is_dead'))['ticker'].to_list()
    last_ret = []
    if dead_names and 'close' in ohlcv.columns:
        lr = (ohlcv.filter(pl.col('ticker').is_in(dead_names)).sort(['ticker', 'date'])
              .with_columns((pl.col('close') / pl.col('close').shift(1).over('ticker') - 1).alias('_r'))
              .group_by('ticker').agg(pl.col('_r').last()))
        last_ret = lr['_r'].drop_nulls().to_list()
    share_crash = float(np.mean(np.array(last_ret) < -0.2)) if last_ret else float('nan')

    out = {
        'n_tickers': n, 'n_dead': n_dead, 'share_dead': n_dead / n if n else float('nan'),
        'years_span': float(years_span), 'delist_rate_per_year': rates, 'delist_rate_median': med_rate,
        'last_return_median': float(np.median(last_ret)) if last_ret else None, 'share_crash': share_crash,
    }
    if universe is not None:
        sz = universe.filter(pl.col('in_universe')).group_by('date').len()['len'].to_numpy()
        out.update({'members_min': int(sz.min()), 'members_median': float(np.median(sz)), 'members_max': int(sz.max())
                    } if len(sz) else {'members_min': 0, 'members_median': 0.0, 'members_max': 0})

    verdict = 'OK'
    if n_dead == 0 and years_span > 3:
        verdict = 'FAIL'
        notes.append('No ticker stops trading before the end of the data. This is the signature of a survivors-only file.')
    if years_span > 10 and n and n_dead / n < 0.10 and n_dead > 0:
        verdict = 'WARN' if verdict == 'OK' else verdict
        notes.append(f'Only {n_dead / n:.0%} of the tickers are dead after {years_span:.0f} years. Expect more in a US all-cap file.')
    if np.isfinite(med_rate) and med_rate < 0.01 and years_span > 3 and verdict != 'FAIL':
        verdict = 'WARN' if verdict == 'OK' else verdict
        notes.append(f'Median yearly delisting rate {med_rate:.1%} is very low.')
    if n_dead > 20 and np.isfinite(share_crash) and share_crash < 0.20:
        verdict = 'WARN' if verdict == 'OK' else verdict
        notes.append(f'Only {share_crash:.0%} of the dead tickers end with a return below -20%. The delisting returns may be missing.')
    if verdict == 'OK':
        notes.append('The file contains dead tickers, at a plausible rate.')
    out['verdict'], out['notes'] = verdict, notes
    return out
