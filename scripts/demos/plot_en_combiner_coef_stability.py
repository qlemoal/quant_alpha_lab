'''
Companion plotting function for coefficient_stability_by_fold(). Not
part of elastic_net_combiner.py itself, kept separate since plotting
dependencies (matplotlib) shouldn't be a hard import for the core module.
'''
import numpy as np
import matplotlib.pyplot as plt
import polars as pl


def plot_coefficient_stability(stability_df: pl.DataFrame, fit_result: dict, panel: pl.DataFrame, save_path: str = None):
    '''
    stability_df: output of coefficient_stability_by_fold().
    fit_result, panel: same objects passed to coefficient_stability_by_fold(),
        needed here to recover each fold's TRAIN date range for the x-axis,
        since CPCV folds aren't in chronological order by construction
        (they're combinatorial), fold INDEX alone doesn't mean "time",
        sorting by each fold's actual train-window midpoint does.
    '''
    panel_dates = panel['date'].to_numpy()
    midpoints = []
    for train_idx, _test_idx in fit_result['cv_folds']:
        fold_dates = panel_dates[train_idx]
        lo, hi = fold_dates.min(), fold_dates.max()
        midpoints.append(lo + (hi - lo) / 2)

    signal_cols = fit_result['signal_cols']
    order = np.argsort(midpoints)
    sorted_midpoints = [midpoints[i] for i in order]

    fig, ax = plt.subplots(figsize=(11, 5))
    for col in signal_cols:
        values = stability_df[col].to_numpy()[order]
        ax.plot(sorted_midpoints, values, 'o', marker='o', markersize=10, label=col, alpha=0.5)
    ax.axhline(0, color='grey', linewidth=0.8, linestyle='--')
    ax.set_xlabel("fold's train-window midpoint (chronological, not fold index)")
    ax.set_ylabel('fitted coefficient')
    ax.set_title('Coefficient stability across CPCV folds, by fold time, not fold index')
    ax.legend()
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()
    if save_path:
        plt.savefig(save_path, dpi=120)
        print('Saved')
    return fig


if __name__ == '__main__':
    import sys, datetime
    sys.path.insert(0, '/home/claude/quant_alpha_lab')
    from src.models.elastic_net_combiner import run_en_combiner, coefficient_stability_by_fold

    N_DATES, N_TICKERS, NOISE_STD = 10000, 30, 3.0
    rng = np.random.default_rng(0)
    calendar_dates = [datetime.date(2005, 1, 1) + datetime.timedelta(days=i) for i in range(N_DATES)]

    rows_date, rows_ticker, rows_open, rows_good, rows_noise = [], [], [], [], []
    for tkr in range(N_TICKERS):
        good_signal = rng.normal(0, 1, N_DATES)
        true_coef_path = np.linspace(0.3, 1.2, N_DATES)  # genuinely drifts over time, the point of this demo
        open_logret = np.zeros(N_DATES)
        open_logret[2:] = true_coef_path[:-2] * good_signal[:-2] + rng.normal(0, NOISE_STD, N_DATES - 2)
        open_price = 100 * np.exp(np.cumsum(open_logret))
        rows_date.extend(calendar_dates)
        rows_ticker.extend([f'T{tkr:02d}'] * N_DATES)
        rows_open.extend(open_price)
        rows_good.extend(good_signal)
        rows_noise.extend(rng.normal(0, 1, N_DATES))

    lf = pl.DataFrame({'date': rows_date, 'ticker': rows_ticker, 'open': rows_open,
                        'good_signal': rows_good, 'noise_signal': rows_noise}).lazy()

    result = run_en_combiner(lf, ['good_signal', 'noise_signal'], horizon=1, embargo_window=16,
                              q_values={'good_signal': 0.01, 'noise_signal': 0.6}, holdout_dates=200)
    stability = coefficient_stability_by_fold(result, result['search_panel'])
    plot_coefficient_stability(stability, result, result['search_panel'],
                                save_path=None)