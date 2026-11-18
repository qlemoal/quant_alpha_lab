'''
Plumbing for stage 02 (prices ETL). The script 02_etl_prices.py only calls these functions.

WHAT THE STAGE DOES

    data/raw/.../ohlcv/AAPL.csv, MSFT.csv, ...      one CSV per ticker, as downloaded by 01
          |
          |  [1] read_raw_prices      all CSVs into one table; every column read as text first
          |  [2] build_clean_table    type the columns, drop the bad rows, remove duplicates, add open_adj
          |  [3] quality_report       count what was dropped and what looks suspicious
          v
    ohlcv.parquet   ticker, date, open, high, low, close_raw, close, volume, open_adj    sorted by ticker, date

THE COLUMNS (names are used by all later stages)
    open, high, low   as traded (split-adjusted by Yahoo, NOT dividend-adjusted)
    close_raw         the close as traded                         (Yahoo "Close")
    close             the ADJUSTED close: splits and dividends    (Yahoo "Adj Close")
    volume            shares traded (split-adjusted by Yahoo)
    open_adj          the open on the same adjusted scale as `close`:

                          open_adj = open * close / close_raw

                      Why: we trade at the OPEN of the next day. The open must be on the same scale
                      as the adjusted close, or the return from one open to the next would jump
                      at every dividend.

NOTHING IS DROPPED IN SILENCE: every removed row is counted by reason in the report.
'''

from pathlib import Path

import duckdb

RAW_COLUMNS = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']


def read_raw_prices(con: "duckdb.DuckDBPyConnection", raw_dir: Path) -> int:
    '''
    Read every CSV of raw_dir into the table `raw` (all columns as text, plus the file name).
    Reading as text first means a bad cell becomes a counted bad row, not a crash.
    Returns the number of raw rows. Raises if there is no CSV or a column is missing.
    '''
    raw_dir = Path(raw_dir)
    if not list(raw_dir.glob('*.csv')):
        raise FileNotFoundError(f'No CSV file in {raw_dir}. Run 01_download_data.py first.')
    con.execute(f'''
        CREATE OR REPLACE TABLE raw AS
        SELECT * FROM read_csv('{raw_dir}/*.csv', filename=true, all_varchar=true, union_by_name=true)
    ''')
    columns = {row[0] for row in con.execute('DESCRIBE raw').fetchall()}
    missing = [c for c in RAW_COLUMNS if c not in columns]
    if missing:
        raise ValueError(f'raw CSV files lack the columns {missing}')
    return con.execute('SELECT count(*) FROM raw').fetchone()[0]


def build_clean_table(con: "duckdb.DuckDBPyConnection") -> dict:
    '''
    Build the table `clean` from `raw`. Returns the number of rows removed, by reason.

    Rules, in this order (a row is counted under the FIRST rule that removes it):
      1. unreadable   a date or a number that cannot be read
      2. not_positive a price <= 0 or a volume <= 0. A day without volume has no real trade.
                      (Its dollar volume would be 0, and log(0) breaks the liquidity features.)
      3. duplicate    several rows for the same (ticker, date). One is kept: the one with most volume.
    '''
    con.execute('''
        CREATE OR REPLACE TABLE typed AS
        SELECT
            regexp_extract(filename, '([^/\\\\]+)\\.csv$', 1)      AS ticker,
            TRY_CAST("Date" AS DATE)                              AS date,
            TRY_CAST("Open" AS DOUBLE)                            AS open,
            TRY_CAST("High" AS DOUBLE)                            AS high,
            TRY_CAST("Low" AS DOUBLE)                             AS low,
            TRY_CAST("Close" AS DOUBLE)                           AS close_raw,
            TRY_CAST("Adj Close" AS DOUBLE)                       AS close,
            TRY_CAST(TRY_CAST("Volume" AS DOUBLE) AS BIGINT)      AS volume
        FROM raw
    ''')
    n_typed = con.execute('SELECT count(*) FROM typed').fetchone()[0]

    con.execute('''
        CREATE OR REPLACE TABLE readable AS
        SELECT * FROM typed
        WHERE date IS NOT NULL AND open IS NOT NULL AND high IS NOT NULL AND low IS NOT NULL
          AND close_raw IS NOT NULL AND close IS NOT NULL AND volume IS NOT NULL
    ''')
    n_readable = con.execute('SELECT count(*) FROM readable').fetchone()[0]

    con.execute('''
        CREATE OR REPLACE TABLE positive AS
        SELECT * FROM readable
        WHERE open > 0 AND high > 0 AND low > 0 AND close_raw > 0 AND close > 0 AND volume > 0
    ''')
    n_positive = con.execute('SELECT count(*) FROM positive').fetchone()[0]

    con.execute('''
        CREATE OR REPLACE TABLE clean AS
        SELECT ticker, date, open, high, low, close_raw, close, volume,
               CAST(open * close / close_raw AS DOUBLE) AS open_adj
        FROM positive
        QUALIFY row_number() OVER (PARTITION BY ticker, date ORDER BY volume DESC, close_raw DESC) = 1
        ORDER BY ticker, date
    ''')
    n_clean = con.execute('SELECT count(*) FROM clean').fetchone()[0]

    return {
        'unreadable': n_typed - n_readable,
        'not_positive': n_readable - n_positive,
        'duplicate': n_positive - n_clean,
    }


def quality_report(con: "duckdb.DuckDBPyConnection", min_history: int = 252) -> dict:
    '''
    Numbers about the clean table. These are WARNINGS for you to read, nothing is removed here.

      high_below_low        high < low                                 (impossible: a data error)
      open_outside_range    open above high or below low, tolerance 0.1%
      adj_factor_above_one  close / close_raw > 1.001. Adjustments for splits and dividends make this
                            ratio <= 1 in the past. A larger value points to bad adjustment data.
      short_history         tickers with fewer than `min_history` rows
    '''
    row = con.execute(f'''
        SELECT
            count(*)                                                   AS n_rows,
            count(DISTINCT ticker)                                     AS n_tickers,
            count(DISTINCT date)                                       AS n_dates,
            min(date)                                                  AS first_date,
            max(date)                                                  AS last_date,
            sum(CASE WHEN high < low THEN 1 ELSE 0 END)                AS high_below_low,
            sum(CASE WHEN open > high * 1.001 OR open < low * 0.999 THEN 1 ELSE 0 END) AS open_outside_range,
            sum(CASE WHEN close / close_raw > 1.001 THEN 1 ELSE 0 END) AS adj_factor_above_one
        FROM clean
    ''').fetchone()
    names = ['n_rows', 'n_tickers', 'n_dates', 'first_date', 'last_date',
             'high_below_low', 'open_outside_range', 'adj_factor_above_one']
    report = dict(zip(names, row))
    report['short_history'] = con.execute(f'''
        SELECT count(*) FROM (SELECT ticker FROM clean GROUP BY ticker HAVING count(*) < {min_history})
    ''').fetchone()[0]
    return report


def write_parquet(con: "duckdb.DuckDBPyConnection", out_file: Path) -> None:
    '''Write the table `clean` (already sorted by ticker and date) to a parquet file.'''
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    con.execute(f"COPY (SELECT * FROM clean ORDER BY ticker, date) TO '{out_file}' (FORMAT PARQUET)")