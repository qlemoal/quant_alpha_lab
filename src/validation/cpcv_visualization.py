'''
Visualizes CPCV fold geometry across a few parameter settings, and sweeps the good_signal/noise_signal recovery check across those same settings.
Not a test, not merged code, exploratory only.
'''


import numpy as np
import matplotlib.pyplot as plt
from src.validation.cpcv import cpcv

N_DATES = 60

settings = [
    dict(n_blocks=6, n_test_blocks=1, purge_w=2, embargo_w=2, title='n_blocks=6, n_test_blocks=1 (true k-fold)'),
    dict(n_blocks=6, n_test_blocks=2, purge_w=2, embargo_w=2, title='n_blocks=6, n_test_blocks=2 (CPCV)'),
    dict(n_blocks=10, n_test_blocks=2, purge_w=2, embargo_w=2, title='n_blocks=10, n_test_blocks=2 (CPCV, more blocks)'),
    dict(n_blocks=10, n_test_blocks=3, purge_w=2, embargo_w=2, title='n_blocks=10, n_test_blocks=3 (CPCV, wider test)'),
]

fig, axes = plt.subplots(len(settings), 1, figsize=(14, 3 * len(settings)))

for ax, s in zip(axes, settings):
    folds = list(cpcv(N_DATES, s['n_blocks'], s['n_test_blocks'], s['purge_w'], s['embargo_w']))
    grid = np.zeros((len(folds), N_DATES))  # 0=train, 1=test, -1=excluded (purge/embargo)
    for i, (train_idx, test_idx, _test_ids) in enumerate(folds):
        grid[i, :] = -1
        grid[i, train_idx] = 0
        grid[i, test_idx] = 1
    im = ax.imshow(grid, aspect='auto', cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_title(f"{s['title']}  ->  {len(folds)} combinations")
    ax.set_xlabel('date index')
    ax.set_ylabel('combination #')

plt.tight_layout()
plt.show()
# plt.savefig('/tmp/cpcv_geometry.png', dpi=120); print('saved geometry plot')




# --- signal recovery sweep across the same settings, on synthetic panel ---

import polars as pl
import datetime
from src.models.elastic_net_combiner import build_design_matrix, apply_q_value_weighting
from sklearn.linear_model import ElasticNetCV

N_DATES_PANEL = 700
N_TICKERS = 30
TRUE_COEF = 0.8

rng = np.random.default_rng(0)
calendar_dates = [datetime.date(2015, 1, 1) + datetime.timedelta(days=i) for i in range(N_DATES_PANEL)]
dates = [d for d in calendar_dates for _ in range(N_TICKERS)]
tickers = np.tile([f'T{i:02d}' for i in range(N_TICKERS)], N_DATES_PANEL)
good_signal = rng.normal(0, 1, N_DATES_PANEL * N_TICKERS)
noise_signal = rng.normal(0, 1, N_DATES_PANEL * N_TICKERS)
fwdret = TRUE_COEF * good_signal + rng.normal(0, 3.0, N_DATES_PANEL * N_TICKERS)
lf = pl.DataFrame({'date': dates, 'ticker': tickers, 'good_signal': good_signal, 'noise_signal': noise_signal, 'fwdret': fwdret}).lazy()

signal_cols = ['good_signal', 'noise_signal']
panel = build_design_matrix(lf, signal_cols, fwd_ret_col='fwdret')
unique_dates = panel['date'].unique().sort().to_numpy()
panel_dates = panel['date'].to_numpy()
X = panel.select(signal_cols).to_numpy()
y = panel['fwdret'].to_numpy()

print(f"\n{'setting':<55} {'n_folds':>8} {'good_signal':>13} {'noise_signal':>13}")
for s in settings:
    n_dates = len(unique_dates)
    row_folds = []
    for train_date_idx, test_date_idx, _ in cpcv(n_dates, s['n_blocks'], s['n_test_blocks'], s['purge_w'], s['embargo_w']):
        train_dates = unique_dates[train_date_idx]
        test_dates = unique_dates[test_date_idx]
        train_row_idx = np.flatnonzero(np.isin(panel_dates, train_dates))
        test_row_idx = np.flatnonzero(np.isin(panel_dates, test_dates))
        row_folds.append((train_row_idx, test_row_idx))

    model = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1], cv=row_folds, n_jobs=-1)
    model.fit(X, y)
    print(f"{s['title']:<55} {len(row_folds):>8} {model.coef_[0]:>13.4f} {model.coef_[1]:>13.4f}")