'''
GOAL: the mean IC of each signal, year by year.

WHY
    One average IC over ten years can hide a signal that worked in two years and failed in eight,
    or a signal that died after a date. The year table shows both.
    It is also a first test of SURVIVORSHIP BIAS: with a universe of today's survivors, old years
    look too good. An IC that is large in the early years and fades toward today is the typical sign.

READING IT
    ic_YYYY          mean rank IC of the signal in that calendar year (raw, not sign-aligned)
    years_ok         number of years where the IC has the SAME sign as the prior (or the overall sign if no prior)
    share_ok         years_ok / years with data
    first_half, second_half  mean aligned IC in the first and second half of the years: a big drop is a warning

This file only reads IC series that stage 05 already computed. It does not touch prices.
'''

import numpy as np
import polars as pl

MIN_DAYS_PER_YEAR = 40        # a year with fewer IC dates is skipped (partial year at the edge of the sample)


def yearly_ic_table(ic_by_signal: dict, signs: dict) -> pl.DataFrame:
    '''
    ic_by_signal: {signal: (dates, ic_values)}   dates = list of datetime.date / pl.Date values, ic_values = array
    signs:        {signal: +1 / -1 / None}       prior sign (None: use the sign of the overall mean)
    Returns one row per signal: signal, one column ic_<year> per year, years_ok, n_years, share_ok,
    first_half, second_half (mean aligned IC of the first and of the second half of the years).
    '''
    rows, all_years = [], set()
    for sig, (dates, ic) in ic_by_signal.items():
        df = pl.DataFrame({'date': dates, 'ic': np.asarray(ic, dtype=float)}).drop_nulls().filter(pl.col('ic').is_finite())
        df = df.with_columns(pl.col('date').dt.year().alias('year'))
        g = df.group_by('year').agg(pl.col('ic').mean().alias('m'), pl.len().alias('n')).filter(pl.col('n') >= MIN_DAYS_PER_YEAR).sort('year')
        years, means = g['year'].to_list(), g['m'].to_numpy()
        sign = signs.get(sig)
        if sign not in (1, -1):
            sign = 1 if (len(means) == 0 or np.nanmean(means) >= 0) else -1
        aligned = means * sign
        half = len(means) // 2
        row = {'signal': sig,
               'years_ok': int((aligned > 0).sum()), 'n_years': len(means),
               'share_ok': float((aligned > 0).mean()) if len(means) else float('nan'),
               'first_half': float(aligned[:half].mean()) if half >= 1 else float('nan'),
               'second_half': float(aligned[half:].mean()) if len(means) - half >= 1 else float('nan')}
        for y, m in zip(years, means):
            row[f'ic_{y}'] = float(m)
            all_years.add(y)
        rows.append(row)
    cols = ['signal'] + [f'ic_{y}' for y in sorted(all_years)] + ['years_ok', 'n_years', 'share_ok', 'first_half', 'second_half']
    full = [{c: r.get(c) for c in cols} for r in rows]
    schema = {c: (pl.Utf8 if c == 'signal' else pl.Int64 if c in ('years_ok', 'n_years') else pl.Float64) for c in cols}
    return pl.DataFrame(full, schema=schema)
