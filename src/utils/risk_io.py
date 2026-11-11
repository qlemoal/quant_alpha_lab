'''
Input/output helpers for the risk model (stage 04d).

FILE FORMAT (unchanged from the previous version)
-------------------------------------------------
A symmetric N x N matrix is stored in LONG format, UPPER TRIANGLE ONLY (i <= j):

    date        ticker_i   ticker_j   <value_name>
    2026-04-29  AAPL       AAPL       1.00
    2026-04-29  AAPL       MSFT       0.55
    ...

Why upper triangle only: the matrix is symmetric, so the lower half carries no new information.
This halves the file size. Use load_matrix_for_date() to get the full N x N matrix back.

Dates are stored as ISO strings ('YYYY-MM-DD'), like before.

PER-DATE FILES, THEN ONE COMBINED FILE
--------------------------------------
Each rebalance date is first written to its own small file ('cov_2026-04-29.parquet').
This lets several worker processes write in parallel without touching the same file.
At the end, combine_parquet_files() glues the small files into one ('cov.parquet') and deletes them.
The gluing is done in streaming mode, so the full history never has to fit in memory.
'''

from pathlib import Path

import numpy as np
import polars as pl


def _per_date_path(file_path: Path, date_str: str) -> Path:
    '''corr_clean.parquet + 2026-04-29 -> corr_clean_2026-04-29.parquet'''
    return file_path.with_stem(f'{file_path.stem}_{date_str}')


# =============================================================================
# WRITE
# =============================================================================

def save_long_matrix(matrix: np.ndarray, tickers: list, date_str: str, file_path: Path, value_name: str) -> Path:
    '''
    Save a symmetric matrix as long format, upper triangle (i <= j), for ONE date.
    Fully vectorised (no Python loop over pairs).

    matrix:     (N, N) array. Row/column k belongs to tickers[k].
    file_path:  the FINAL combined path. The per-date file name is derived from it.
    Returns the per-date file path.
    '''
    matrix = np.asarray(matrix, dtype=float)
    n = len(tickers)
    if matrix.shape != (n, n):
        raise ValueError(f'matrix shape {matrix.shape} does not match {n} tickers (date {date_str})')

    names = np.asarray(tickers)
    i, j = np.triu_indices(n)                      # all pairs with i <= j
    df = pl.DataFrame({
        'ticker_i': names[i],
        'ticker_j': names[j],
        value_name: matrix[i, j],
    }).with_columns(pl.lit(date_str).alias('date')).select(['date', 'ticker_i', 'ticker_j', value_name])

    out = _per_date_path(file_path, date_str)
    df.write_parquet(out)
    return out


def save_clusters(cluster_map: dict, valid_tickers: list, date_str: str, file_path: Path) -> Path:
    '''
    Save {ticker_index: cluster_id} as rows [date, ticker, cluster_id], for ONE date.
    A ticker missing from cluster_map is NOT written (the caller should check it, see risk_pipeline).
    '''
    rows = {
        'date': [date_str] * len(cluster_map),
        'ticker': [valid_tickers[t] for t in cluster_map],
        'cluster_id': [int(c) for c in cluster_map.values()],
    }
    out = _per_date_path(file_path, date_str)
    pl.DataFrame(rows, schema={'date': pl.String, 'ticker': pl.String, 'cluster_id': pl.Int64}).write_parquet(out)
    return out


def save_eigenvalues(eigenvals: np.ndarray, date_str: str, file_path: Path) -> Path:
    '''Save the eigenvalues of the cleaned correlation, for ONE date. Rows: [date, eigenvalue, component].'''
    eigenvals = np.asarray(eigenvals, dtype=float)
    df = pl.DataFrame({
        'date': [date_str] * len(eigenvals),
        'eigenvalue': eigenvals,
        'component': np.arange(len(eigenvals)),
    })
    out = _per_date_path(file_path, date_str)
    df.write_parquet(out)
    return out


def combine_parquet_files(output_dir: Path, base_name: str, date_suffixes: list) -> None:
    '''
    Glue the per-date files {base_name}_{date}.parquet into {base_name}.parquet, then delete them.

    Streaming (sink_parquet): the rows go from the small files to the big file chunk by chunk.
    If streaming is not possible for the installed polars version, we fall back to an in-memory concat.
    The small files are deleted ONLY after the combined file is fully written.
    '''
    output_dir = Path(output_dir)
    files = [output_dir / f'{base_name}_{ds}.parquet' for ds in date_suffixes]
    files = [f for f in files if f.exists()]
    if not files:
        return

    out = output_dir / f'{base_name}.parquet'
    try:
        pl.concat([pl.scan_parquet(f) for f in files]).sink_parquet(out)
    except Exception:
        pl.concat([pl.read_parquet(f) for f in files]).write_parquet(out)

    for f in files:
        f.unlink()


# =============================================================================
# READ
# =============================================================================

def load_matrix_for_date(file_path: Path, date_str: str, value_name: str) -> "tuple[np.ndarray, list]":
    '''
    Read one date of a long-format matrix file and rebuild the FULL symmetric N x N matrix.

    Returns (matrix, tickers). `tickers` is sorted alphabetically and gives the row/column order.
    This is the function to use in 04e / 04f instead of pivoting the long table by hand
    (a pivot of the upper triangle alone would leave the lower half empty).
    '''
    df = pl.scan_parquet(file_path).filter(pl.col('date') == date_str).collect()
    if df.height == 0:
        raise ValueError(f'no rows for date {date_str} in {file_path}')

    tickers = sorted(set(df['ticker_i'].to_list()) | set(df['ticker_j'].to_list()))
    index = {t: k for k, t in enumerate(tickers)}
    i = np.array([index[t] for t in df['ticker_i'].to_list()])
    j = np.array([index[t] for t in df['ticker_j'].to_list()])
    v = df[value_name].to_numpy()

    n = len(tickers)
    matrix = np.zeros((n, n))
    matrix[i, j] = v
    matrix[j, i] = v
    return matrix, tickers