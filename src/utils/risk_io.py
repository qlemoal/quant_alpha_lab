import polars as pl
from pathlib import Path
import numpy as np


def save_long_matrix(matrix: np.ndarray, tickers: list, date_str: str, file_path: Path, value_name: str) -> Path:
    '''
    Save symmetric matrix in long format (one row per i,j pair).
    Returns the temporary file path for per-date storage.
    '''
    n = len(tickers)
    rows = []
    for i in range(n):
        for j in range(i, n):
            rows.append({
                'date': date_str,
                'ticker_i': tickers[i],
                'ticker_j': tickers[j],
                value_name: float(matrix[i, j])
            })
    df = pl.DataFrame(rows)
    temp_path = file_path.with_stem(f'{file_path.stem}_{date_str}')
    df.write_parquet(temp_path)
    return temp_path


def save_clusters(cluster_map: dict, valid_tickers: list, date_str: str, file_path: Path) -> Path:
    rows = [{'date': date_str, 'ticker': valid_tickers[t], 'cluster_id': int(c)} for t, c in cluster_map.items()]
    df = pl.DataFrame(rows)
    temp_path = file_path.with_stem(f'{file_path.stem}_{date_str}')
    df.write_parquet(temp_path)
    return temp_path


def combine_parquet_files(output_dir: Path, base_name: str, date_suffixes: list) -> None:
    files = [output_dir / f'{base_name}_{ds}.parquet' for ds in date_suffixes if (output_dir / f'{base_name}_{ds}.parquet').exists()]
    if not files:
        return
    dfs = [pl.read_parquet(f) for f in files]
    combined = pl.concat(dfs)
    combined.write_parquet(output_dir / f'{base_name}.parquet')
    for f in files:
        f.unlink()


def save_eigenvalues(eigenvals: np.ndarray, date_str: str, file_path: Path) -> Path:
    rows = [{'date': date_str, 'eigenvalue': float(v), 'component': i} for i, v in enumerate(eigenvals)]
    df = pl.DataFrame(rows)
    temp_path = file_path.with_stem(f'{file_path.stem}_{date_str}')
    df.write_parquet(temp_path)
    return temp_path