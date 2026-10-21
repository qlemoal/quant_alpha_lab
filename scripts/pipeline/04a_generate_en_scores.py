#!/usr/bin/env python
'''
Phase 0: Generate Elastic Net Combined Scores
===============================================

Runs the Elastic Net combiner to produce combined_score from candidate signals.
Run this AFTER 03_build_features.py and signal evaluation/FDR.

Outputs:
  data/processed/en_scores.parquet   # [date, ticker, combined_score, <other_cols>]

Usage:
  python scripts/pipeline/04a_generate_en_scores.py

Env vars:
  EN_ALPHA              = 'auto'  # or float, 'auto' uses CPCV search
  EN_L1_RATIO           = 'auto'  # or float
  EN_CV_METHOD          = 'cpcv'  # 'cpcv' or 'walkforward'
  EN_Q_VALUE_WEIGHTING  = 'true'  # apply (1-q) column weighting
  EN_MIN_IC_IR          = '0.0'   # minimum IR to include signal
  SCORE_COL             = 'combined_score'
'''

import os
import sys
from pathlib import Path

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.elastic_net_combiner import fit_elastic_net_combiner
from src.evaluation.signals.fdr import fdr_report
from src.evaluation.signals.ic.core import compute_ic, summarize_ic
import polars as pl

# =============================================================================
# CONFIGURATION
# =============================================================================

# Input
FEATURES_FILE = PROJECT_ROOT / 'data' / 'processed' / 'features.parquet'
SIGNALS_FILE = PROJECT_ROOT / 'data' / 'processed' / 'signals.parquet'  # from make_signal()
PRICES_FILE = PROJECT_ROOT / 'data' / 'processed' / 'prices.parquet'

# Output
OUTPUT_FILE = PROJECT_ROOT / 'data' / 'processed' / 'en_scores.parquet'

# EN Configuration
EN_ALPHA = os.environ.get('EN_ALPHA', 'auto')
EN_L1_RATIO = os.environ.get('EN_L1_RATIO', 'auto')
EN_CV_METHOD = os.environ.get('EN_CV_METHOD', 'cpcv')
EN_Q_VALUE_WEIGHTING = os.environ.get('EN_Q_VALUE_WEIGHTING', 'true').lower() == 'true'
EN_MIN_IC_IR = float(os.environ.get('EN_MIN_IC_IR', '0.0'))
SCORE_COL = os.environ.get('SCORE_COL', 'combined_score')

# CV Configuration (if walkforward)
WF_N_FOLDS = int(os.environ.get('WF_N_FOLDS', '5'))
WF_EMBARGO = int(os.environ.get('WF_EMBARGO', '37'))

# CPCV Configuration
CPCV_N_BLOCKS = int(os.environ.get('CPCV_N_BLOCKS', '12'))
CPCV_N_TEST_BLOCKS = int(os.environ.get('CPCV_N_TEST_BLOCKS', '2'))
CPCV_EMBARGO = int(os.environ.get('CPCV_EMBARGO', '37'))

# Forward return horizon
HORIZON = int(os.environ.get('HORIZON', '20'))
FWD_RET_COL = f'fwd_ret_{HORIZON}d'

# Date range for training
TRAIN_START = os.environ.get('TRAIN_START', '2010-01-01')
TRAIN_END = os.environ.get('TRAIN_END', '2024-12-31')


def load_candidate_signals():
    '''Load and filter candidate signals from signals.parquet'''
    print(f'Loading candidate signals from {SIGNALS_FILE}...')

    if not SIGNALS_FILE.exists():
        raise FileNotFoundError(f'Signals file not found: {SIGNALS_FILE}. Run signal generation first.')

    signals_lf = pl.scan_parquet(SIGNALS_FILE)

    # Get all signal columns (exclude date, ticker, forward returns)
    schema = signals_lf.collect_schema()
    all_cols = schema.names()

    # Identify signal columns (numeric, not date/ticker/fwd_ret)
    exclude = {'date', 'ticker', 'fwd_ret_1d', 'fwd_ret_5d', 'fwd_ret_10d', 'fwd_ret_20d'}
    signal_cols = [c for c in all_cols if c not in exclude and schema[c].is_numeric()]

    print(f'  Found {len(signal_cols)} candidate signals: {signal_cols[:10]}{"..." if len(signal_cols) > 10 else ""}')

    # Load with forward returns
    required = ['date', 'ticker'] + signal_cols + [FWD_RET_COL]
    missing = [c for c in required if c not in all_cols]
    if missing:
        raise ValueError(f'Missing columns in signals file: {missing}')

    df = signals_lf.select(required).filter(
        (pl.col('date') >= TRAIN_START) & (pl.col('date') <= TRAIN_END)
    ).collect()

    print(f'  Loaded {len(df):,} rows, {len(signal_cols)} signals')
    return df, signal_cols


def filter_signals_by_ic(df, signal_cols):
    '''Filter signals by minimum IC IR using FDR'''
    print(f'\nFiltering signals by IC IR >= {EN_MIN_IC_IR}...')

    ic_results = []
    for col in signal_cols:
        ic_series = compute_ic(df, col, FWD_RET_COL)
        summary = summarize_ic(ic_series)
        ic_results.append({
            'signal': col,
            'mean_ic': summary['mean'],
            'std_ic': summary['std'],
            'ir': summary['ir'],
            'hit_rate': summary['hit_rate'],
            'n_obs': summary['n']
        })

    ic_df = pl.DataFrame(ic_results)
    print(f'  IC Summary:')
    for row in ic_df.iter_rows(named=True):
        print(f'    {row["signal"]:20s}  IC={row["mean_ic"]:6.4f}  IR={row["ir"]:6.3f}  HR={row["hit_rate"]:.2%}')

    # FDR on p-values from IC t-stats
    from scipy import stats
    ic_df = ic_df.with_columns([
        (2 * (1 - stats.norm.cdf(abs(pl.col('ir') * (pl.col('n_obs') ** 0.5))))).alias('p_value')
    ])

    fdr_result = fdr_report(ic_df.to_pandas(), pval_col='p_value', alpha=0.2)
    print(f'\n  FDR (BY, alpha=0.2): {fdr_result["n_survive_by"]}/{len(ic_df)} survive')
    print(f'  FDR (q-value):       {fdr_result["n_survive_q"]}/{len(ic_df)} survive (q<0.2)')

    # Filter
    survived = fdr_result[fdr_result['survives_q']]['signal'].tolist()
    if EN_MIN_IC_IR > 0:
        survived = [s for s in survived if ic_df.filter(pl.col('signal') == s)['ir'][0] >= EN_MIN_IC_IR]

    print(f'  Final candidates: {survived}')
    return survived, ic_df, fdr_result


def main():
    print('=' * 70)
    print('PHASE 0: GENERATE ELASTIC NET SCORES')
    print('=' * 70)
    print(f'Train period:   {TRAIN_START} to {TRAIN_END}')
    print(f'Forward horizon: {HORIZON}d ({FWD_RET_COL})')
    print(f'CV method:      {EN_CV_METHOD}')
    print(f'Q-value weighting: {EN_Q_VALUE_WEIGHTING}')
    print(f'Output:         {OUTPUT_FILE}')
    print('=' * 70)

    # Load signals
    df, signal_cols = load_candidate_signals()

    # Filter by IC/FDR
    survived_signals, ic_df, fdr_result = filter_signals_by_ic(df, signal_cols)

    if len(survived_signals) < 2:
        print('\n⚠️  WARNING: Fewer than 2 signals survived FDR. EN needs at least 2.')
        print('    Proceeding with all signals (no FDR filter)...')
        survived_signals = signal_cols

    # Fit Elastic Net
    print(f'\nFitting Elastic Net with {len(survived_signals)} signals...')
    print(f'  Alpha: {EN_ALPHA}, L1_ratio: {EN_L1_RATIO}')
    print(f'  Q-value weighting: {EN_Q_VALUE_WEIGHTING}')

    # Get q-values for survived signals
    q_values = {}
    for _, row in fdr_result.iterrows():
        if row['signal'] in survived_signals:
            q_values[row['signal']] = row['q_value']

    # Prepare data for EN
    X = df.select(survived_signals).to_numpy()
    y = df.select(FWD_RET_COL).to_numpy().ravel()

    # Handle missing
    import numpy as np
    mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X = X[mask]
    y = y[mask]
    dates = df.select('date').to_numpy().ravel()[mask]
    tickers = df.select('ticker').to_numpy().ravel()[mask]

    print(f'  After dropping NaN: {len(X):,} samples')

    # Fit
    if EN_CV_METHOD == 'cpcv':
        from src.validation.cpcv import purged_embargoed_kfold_splits
        from sklearn.model_selection import GridSearchCV
        from sklearn.linear_model import ElasticNet
        import numpy as np

        # Create CPCV splits on row indices
        n_samples = len(X)
        # Simple approach: use date-based blocks
        unique_dates = sorted(set(dates))
        n_dates = len(unique_dates)
        dates_array = np.array(dates)

        # For now, use walk-forward as fallback if CPCV complex
        print('  Using walk-forward CV (CPCV on pooled panel not yet wired)')
        cv_method = 'walkforward'
    else:
        cv_method = 'walkforward'

    # Use the existing fit function
    from src.models.elastic_net_combiner import fit_elastic_net_combiner

    # Build design matrix DataFrame for the combiner
    en_input = df.select(['date', 'ticker'] + survived_signals + [FWD_RET_COL])

    result = fit_elastic_net_combiner(
        signals_df=en_input,
        signal_cols=survived_signals,
        fwd_ret_col=FWD_RET_COL,
        alpha=EN_ALPHA if EN_ALPHA != 'auto' else None,
        l1_ratio=EN_L1_RATIO if EN_L1_RATIO != 'auto' else None,
        cv_method=cv_method,
        n_folds=WF_N_FOLDS,
        embargo=WF_EMBARGO,
        q_values=q_values if EN_Q_VALUE_WEIGHTING else None,
    )

    # Extract predictions
    print('\nGenerating combined scores...')
    coef = result['coefficients']
    intercept = result['intercept']
    alpha_used = result['alpha']
    l1_ratio_used = result['l1_ratio']

    print(f'  Alpha: {alpha_used:.6f}, L1_ratio: {l1_ratio_used:.3f}')
    print(f'  Non-zero coefficients: {(np.abs(coef) > 1e-6).sum()}/{len(coef)}')
    for name, c in zip(survived_signals, coef):
        if abs(c) > 1e-6:
            print(f'    {name:20s}: {c:8.6f}')

    # Compute scores for ALL data (not just train)
    print('\nScoring full dataset...')
    full_df = pl.scan_parquet(SIGNALS_FILE).collect()
    X_full = full_df.select(survived_signals).to_numpy()

    # Handle missing in full data - impute with 0 (median of normalized)
    X_full = np.nan_to_num(X_full, nan=0.0)
    scores = X_full @ coef + intercept

    # Add to full dataframe
    full_df = full_df.with_columns(pl.Series(name=SCORE_COL, values=scores))

    # Save
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    full_df.select(['date', 'ticker', SCORE_COL] + survived_signals).write_parquet(OUTPUT_FILE)

    print(f'\n✓ Saved {len(full_df):,} rows to {OUTPUT_FILE}')
    print(f'  Columns: date, ticker, {SCORE_COL}, + {len(survived_signals)} signal cols')

    # Save model info
    import json
    model_info = {
        'alpha': float(alpha_used),
        'l1_ratio': float(l1_ratio_used),
        'signals': survived_signals,
        'coefficients': {k: float(v) for k, v in zip(survived_signals, coef)},
        'intercept': float(intercept),
        'cv_method': cv_method,
        'q_weighting': EN_Q_VALUE_WEIGHTING,
        'train_start': TRAIN_START,
        'train_end': TRAIN_END,
        'horizon': HORIZON,
        'fdr_result': fdr_result[['signal', 'p_value', 'q_value', 'survives_q']].to_dict('records')
    }
    model_file = OUTPUT_FILE.with_suffix('.json')
    with open(model_file, 'w') as f:
        json.dump(model_info, f, indent=2, default=str)
    print(f'  Model info: {model_file}')

    print('\nDone!')
    print(f'\nNext: python scripts/pipeline/04b_estimate_risk_model.py')


if __name__ == '__main__':
    main()