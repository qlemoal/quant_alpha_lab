'''
Returns: the past return of each day (a FEATURE) and the future returns (LABELS).

FEATURE versus LABEL: the rule that matters most in this file
-------------------------------------------------------------
    logret   uses the close of day t and of day t-1.       Known at the close of t.  A FEATURE.
    fwdret*  use prices AFTER day t.                        Unknown at t.             A LABEL.

    A label is only for measuring a signal (IC) or training a model against it.
    Never build a signal from a label. Stage 04 rebuilds the labels it needs itself.

THE CLOCK OF THE LABELS (open to open)
--------------------------------------
    day t           day t+1                 day t+1+h
    close: signal   open: we trade (entry)  open: exit
                    |<------ fwdret{h} ---->|

    fwdret{h}[t] = log(open_adj[t+1+h]) - log(open_adj[t+1])

All functions work per ticker. The frame must have the columns ticker and date. Rows are sorted by
ticker and date inside each function. A shift moves by ROWS: it equals trading days only if the ticker
has a row for every trading day (stage 04 reports the missing rows).
'''

import polars as pl
from polars import col as c


def add_log_returns(lf: pl.LazyFrame) -> pl.LazyFrame:
    '''
    logret[t] = log(close[t]) - log(close[t-1]), per ticker. 'close' is the ADJUSTED close
    (splits and dividends), so logret is a total return. The first row of a ticker is null.
    '''
    return lf.sort(['ticker', 'date']).with_columns(
        c('close').log().diff().over('ticker').alias('logret')
    )


def add_fwd_returns(lf: pl.LazyFrame) -> pl.LazyFrame:
    '''
    LABEL (legacy name). One-day forward return with next-day execution:
    fwdret[t] = log(open_adj[t+2]) - log(open_adj[t+1]). Same as add_fwdret_horizon(lf, 1).
    Kept for the research scripts that read the column 'fwdret'.
    '''
    return lf.sort(['ticker', 'date']).with_columns(
        c('open_adj').log().diff().shift(-2).over('ticker').alias('fwdret')
    )


def add_fwdret_horizon(lf: pl.LazyFrame, horizon: int = 1, terminal: bool = False) -> pl.LazyFrame:
    '''
    LABEL. Forward log-return over `horizon` trading days, open to open, with one day of delay:

        fwdret{h}[t] = log(open_adj[t+1+h]) - log(open_adj[t+1])

    The last h+1 rows of each ticker are null: those prices do not exist yet.

    terminal=True (needs a boolean column `is_dead`, see src/universe/point_in_time.py):
        for a DEAD ticker (delisted, merged, bankrupt) the exit price that does not exist is replaced by the
        LAST adjusted close of the ticker. That close already contains the delisting return. Without this,
        the last h+1 labels of every dead stock are null and its final crash is never measured: a survivorship
        bias inside the labels. The entry price t+1 must still exist (no trade is possible after the last row),
        so the very last row keeps a null label. A stock that is still alive at the end of the data keeps nulls.
    '''
    # The shift MUST happen inside the per-ticker window: `.over('ticker')` is applied to the whole
    # expression. (Applied to the log only, the shift runs over the full frame and the last rows of a ticker
    # read the prices of the NEXT ticker: wrong labels, not null, in the last horizon+1 rows of every ticker.)
    log_open = c('open_adj').log()
    exit_ = log_open.shift(-(horizon + 1))
    if terminal:
        exit_ = pl.when(exit_.is_null() & c('is_dead')).then(c('close').log().last()).otherwise(exit_)
    fwd_expr = (exit_ - log_open.shift(-1)).over('ticker')
    return lf.sort(['ticker', 'date']).with_columns(fwd_expr.alias(f'fwdret{horizon}'))
