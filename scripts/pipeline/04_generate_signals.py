#!/usr/bin/env python
'''
===============================================
Phase 04: Generate Signals
===============================================

QUESTION ANSWERED
    "Which candidate signals do we test, and what are the returns they will be tested against?"

THE STAGE IN ONE PICTURE

    features.parquet                       SIGNAL_SPECS (below)            HORIZONS (below)
    (mom20, vol60, log_adv60, open_adj)    (feature, transform, params)    (1, 5, 10, 20 days)
            |                                      |                              |
            v                                      v                              v
     [1] load the features, cut the dates (optional)
            |
            v
     [2] src/signals/signal_pipeline.py
            clean features    NaN and inf -> null
            build signals     one column per spec, normalised across the stocks of the SAME date
            add labels        fwdret{h} = log(open_adj[t+1+h]) - log(open_adj[t+1])
            |
            v
     [3] checks: labels aligned per ticker, no missing dates, signals not empty or constant
            |
            v
     [4] signals.parquet   +   config.json (specs, hashes, git commit, descriptive statistics)

WHERE IS THE COMPLICATED PART?
    src/signals/signal_pipeline.py   cleaning, signals, labels, checks, statistics
    src/signals/combine.py           the transforms (zscore_tanh, zscore_clip, rank, decile)
    This script only lists the signals you want and calls them, in order.

WHAT 04 NEVER DOES
    It never measures how well a signal predicts returns (no IC, no Sharpe, no drawdown).
    That is stage 05, on the search period only. Looking at IC over all dates here would let the
    holdout influence which signals you decide to build. The statistics in config.json use no returns.

THE CLOCK
    The signal of row t uses data up to the close of day t.
    The label of row t is the return from the open of t+1 to the open of t+1+h. It uses only later prices.

BENCHMARK SIGNALS (two options, both for testing the pipeline, never for a strategy)
    --cheat     python scripts/pipeline/04_generate_signals.py --date 2026-11-07 --tag cheat --cheat
                adds signals built from the FUTURE return plus noise, with a known IC (CHEAT_SIGNALS below).
                They show what 05, 06 and 09 do with a signal of a known strength.
                The tag MUST contain the word "cheat".
    --placebo   python scripts/pipeline/04_generate_signals.py --date 2026-11-07 --tag placebo --placebo
                adds PLACEBO_COUNT signals of pure noise (true IC = 0). No return is used.
                A sound pipeline selects none of them, and gives them no weight in 06. The tag MUST contain "placebo".
    Both:       --tag cheat_placebo --cheat --placebo
    See src/signals/cheat.py. Never quote a result of such a run as a strategy result.

HOW TO ADD A SIGNAL
    Add one line to SIGNAL_SPECS: (feature column, transform, parameters). Then run 04, 05, ...
    Every signal you ever try counts as one test for the false discovery rate in 05.
    Keep the list of what you tried, also the ones you removed.

INPUTS    {clean_base}/features/features.parquet
OUTPUTS   {clean_base}/runs/{run_id}/signals/
    signals.parquet   [date, ticker, <signals>, fwdret1, fwdret5, fwdret10, fwdret20]
    config.json       specs, skipped specs, hashes, git commit, statistics, label checks

USAGE
    python scripts/pipeline/04_generate_signals.py --date 2026-11-07 --tag default
'''

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# Make `import src...` and `import config...` work wherever the script is started from.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import polars as pl

from config.paths import get_paths, run_id_from_args
from src.signals.signal_pipeline import (
    validate_specs, clean_features, build_signals, add_labels, restrict_dates, add_dead_flag, apply_universe,
    check_labels, calendar_gaps, signal_descriptives, signal_warnings,
)
from src.signals.cheat import add_cheat_signals, add_placebo_signals, validate_cheat_specs, check_benchmark_tag
from src.utils.helpers import get_git_commit, get_git_diff, file_hash, params_hash
from src.utils import console as out


# =============================================================================
# CONFIGURATION
# =============================================================================

# One line per signal: (feature column, transform, parameters). The signal is named '{feature}_{transform}'.
#   'zscore_tanh'  z-score across the stocks of the date, then tanh.   params: scale (default 1.0)
#   'zscore_clip'  z-score, then clip.                                 params: low, high (default -3, 3)
#   'zscore'       z-score only.
#   'rank'         rank across the stocks, scaled to [-1, 1].          params: descending (default False)
#   'decile'       +1 top decile, -1 bottom decile, 0 otherwise.       params: n_buckets (default 10)
# The z-score transforms take `descending=True` to flip the sign (a feature where LOW is good).
SIGNAL_SPECS = [
    # Momentum: several horizons
    ('mom5',   'zscore_tanh', {}),
    ('mom10',  'zscore_tanh', {}),
    ('mom20',  'zscore_tanh', {}),
    ('mom60',  'zscore_tanh', {}),
    ('mom120', 'zscore_tanh', {}),
    ('mom252', 'zscore_tanh', {}),

    # Volatility (the sign is NOT flipped here: 05 and the combiner learn the sign from the data)
    ('vol5',   'zscore_clip', {'low': -3, 'high': 3}),
    ('vol10',  'zscore_clip', {'low': -3, 'high': 3}),
    ('vol20',  'zscore_clip', {'low': -3, 'high': 3}),
    ('vol60',  'zscore_clip', {'low': -3, 'high': 3}),
    ('vol120', 'zscore_clip', {'low': -3, 'high': 3}),
    ('vol252', 'zscore_clip', {'low': -3, 'high': 3}),

    # Liquidity (average dollar volume, log)
    ('log_adv10', 'zscore_tanh', {}),
    ('log_adv20', 'zscore_tanh', {}),
    ('log_adv60', 'zscore_tanh', {}),

    # Tried and switched off (keep this list: every signal you try counts for the false discovery rate)
    # ('log_adv5',   'zscore_tanh', {}),
    # ('log_adv120', 'zscore_tanh', {}),
    # ('log_adv252', 'zscore_tanh', {}),
    # ('beta60',  'rank', {'descending': True}),
    # ('beta120', 'rank', {'descending': True}),
    # ('beta252', 'rank', {'descending': True}),
]

# Forward-return horizons (trading days). 05 and 06 use ONE of them (default 20).
HORIZONS = [1, 5, 10, 20]

# Optional date cut, 'YYYY-MM-DD' or None. The labels are built after the cut, so the last rows have no label.
DATE_START = None
DATE_END = None

# Cheating signals, added ONLY with the option --cheat. One line per signal: (label horizon, wanted rank IC).
#   IC 0.01 very weak   0.02 weak (a typical real signal)   0.03 and 0.05 good   0.10 strong   0.30 very strong
#   (max 0.99). The horizon must be in HORIZONS. The column name is cheat_h{horizon}_ic{IC x 1000}, e.g. cheat_h20_ic050.
CHEAT_SIGNALS = [
    (20, 0.01),
    (20, 0.02),
    (20, 0.03),
    (20, 0.05),
]
CHEAT_SEED = 0                  # another seed gives another draw of the noise
CHEAT_NOISE_DAYS = 20           # 1 = new noise every day. 20 = noise that changes slowly, like a real signal.
                                # A persistent signal trades less, and its IC series has fewer independent dates.

# Placebo signals, added ONLY with the option --placebo: how many signals of pure noise (placebo_01, ...).
# They count as tests in the false discovery rate, like every signal you try. About 20 is a good size:
# at a 5% level, 1 of them is expected to look significant by chance, and no more.
PLACEBO_COUNT = 20
PLACEBO_SEED = 0
PLACEBO_NOISE_DAYS = [1, 5, 20, 60]   # the placebos use these persistences in turn (fast junk and slow junk)

MAX_MISSING_SHARE = 0.01        # warn if more than 1% of the expected (ticker, date) rows are missing


def main(run_date: str, tag: str, cheat: bool = False, placebo: bool = False):
    start = time.time()
    check_benchmark_tag(tag, cheat, placebo)       # a run with benchmark signals must say so in its tag
    cheat_specs = validate_cheat_specs(CHEAT_SIGNALS, HORIZONS) if cheat else []
    run_id = run_id_from_args(run_date, tag)
    cfg = get_paths(run_id=run_id)
    out_dir = cfg.stage_dir('signals')

    if not cfg.features_file.exists():
        raise FileNotFoundError(f'Missing input: {cfg.features_file}. Run 03_build_features.py first.')

    out.stage('04', 'Generate signals', 'Which candidate signals do we test, and against which future returns?')
    out.info('run', run_id)
    out.info('dataset', f'{cfg.asset}/{cfg.region}/{cfg.universe}/{cfg.freq}')
    out.info('dates', f'{DATE_START or "first"} to {DATE_END or "last"}')
    out.info('label horizons (days)', HORIZONS)
    if cheat:
        out.warn('CHEAT run: the cheat signals use FUTURE returns. A test of the pipeline, never a strategy.')
    if placebo:
        out.note(f'PLACEBO run: {PLACEBO_COUNT} signals of pure noise are added (true IC = 0). A sound pipeline admits none.')

    # ------------------------------------------------------------------
    # [1/4] Load (lazy: nothing is read yet) and validate the specs
    # ------------------------------------------------------------------
    out.step(1, 4, 'Read the features and check the signal list',
             'a signal is a transform of one feature; a spec whose feature is missing is skipped, not guessed')
    lf = restrict_dates(pl.scan_parquet(cfg.features_file), DATE_START, DATE_END)
    available = set(lf.collect_schema().names())
    missing_cols = {'date', 'ticker', 'open_adj'} - available
    if missing_cols:
        raise ValueError(f'features.parquet lacks the columns {sorted(missing_cols)}')

    specs, skipped = validate_specs(SIGNAL_SPECS, available)
    for sk in skipped:
        out.warn(f'skipped {sk["spec"][0]}: {sk["reason"]}')
    if not specs:
        raise ValueError('No signal can be built: none of the SIGNAL_SPECS features is in features.parquet.')
    out.info('signals to build', f'{len(specs)} ({len(skipped)} skipped)')

    # ------------------------------------------------------------------
    # [2/4] Build the signals and the labels (lazy), then compute once
    # ------------------------------------------------------------------
    out.step(2, 4, 'Build the labels, apply the universe, build the signals',
             'labels first, on every row of a ticker (they need its future rows); then keep only the stocks that '
             'were investable on each date; then rank each signal across those stocks only')
    lf = clean_features(lf, sorted({feat for feat, _, _ in specs}))

    # Labels FIRST, on every row of every ticker (a label needs the future rows of the same ticker).
    # A dead ticker keeps its last adjusted close as the exit price, so its final crash is in the labels.
    lf = add_dead_flag(lf)
    lf, label_cols = add_labels(lf, HORIZONS, terminal=True)
    lf_all_rows = lf                                  # kept for the checks: they need the rows BEFORE the universe filter

    # Then the point-in-time universe: a signal is ranked only among the stocks investable on that date.
    universe_info = {'applied': False}
    if cfg.universe_file.exists():
        lf, universe_info = apply_universe(lf, cfg.universe_file)
        universe_info['applied'] = True
        out.ok(f'point-in-time universe applied ({cfg.universe_file.name}): each date uses only its own members')
    else:
        out.warn('no universe file: every ticker is used on every date it trades. With a list of today\'s '
                 'survivors this is SURVIVORSHIP BIAS. Run 02b_build_universe.py (docs/survivorship.md).')

    lf, signal_cols = build_signals(lf, specs)
    cheat_cols = []
    if cheat:
        lf, cheat_cols = add_cheat_signals(lf, cheat_specs, seed=CHEAT_SEED, noise_days=CHEAT_NOISE_DAYS)      # needs the labels, so after them
        signal_cols = signal_cols + cheat_cols
    placebo_cols = []
    if placebo:
        lf, placebo_cols = add_placebo_signals(lf, PLACEBO_COUNT, seed=PLACEBO_SEED, noise_days=PLACEBO_NOISE_DAYS)
        signal_cols = signal_cols + placebo_cols
    real_cols = [c for c in signal_cols if c not in cheat_cols and c not in placebo_cols]
    out.info('real signals', f'{len(real_cols)}: {", ".join(real_cols)}')
    if cheat:
        out.info('cheat signals', ', '.join(cheat_cols))
    if placebo:
        out.info('placebo signals', f'{placebo_cols[0]} ... {placebo_cols[-1]} ({len(placebo_cols)})')
    out.info('labels', ', '.join(label_cols) + '   (open of t+1 to open of t+1+h)')

    output_cols = ['date', 'ticker'] + signal_cols + label_cols
    df = lf.select(output_cols).collect()
    out.info('table', f'{df.height:,} rows, {df["ticker"].n_unique()} tickers, {df["date"].n_unique()} dates '
                      f'({df["date"].min()} to {df["date"].max()})')

    # ------------------------------------------------------------------
    # [3/4] Checks
    # ------------------------------------------------------------------
    out.step(3, 4, 'Check the labels and the signals (no return-based statistic here)',
             'a label that crosses from one ticker to the next, or a missing day, silently corrupts every later '
             'number; IC is NOT computed here, so the holdout cannot influence which signals you build')
    dead_set = set(lf_all_rows.filter(pl.col('is_dead')).select('ticker').unique().collect()['ticker'].to_list())
    slim = lf_all_rows.select(['ticker', 'date'] + label_cols).collect()
    label_report = check_labels(slim, HORIZONS, dead_tickers=dead_set)    # raises if labels cross tickers
    del slim
    out.ok('labels stay inside their ticker (the last h+1 rows of each ticker have no label)')
    out.info('dead tickers', f'{len(dead_set)} (their labels end at the last price, so the delisting crash is in them)')
    out.table(['label', 'null %', 'std', 'max |x|', '|x| > 2'],
              [[col, f'{r["null_pct"]:.2f}', f'{r["std"]:.4f}', f'{r["max_abs"]:.2f}', r['n_abs_above_2']]
               for col, r in label_report.items()], [-10, 7, 7, 8, 8])
    for col, r in label_report.items():
        if r['n_abs_above_2_alive']:
            out.warn(f'{col}: {r["n_abs_above_2_alive"]} extreme labels on stocks that did not die. '
                     'Look for an unadjusted split in the prices.')

    gaps = calendar_gaps(lf_all_rows)
    out.info('missing days inside a life', f'{gaps["missing_rows"]:,} ({gaps["missing_share"]:.2%}); '
             f'worst {gaps["worst_ticker"]} ({gaps["worst_ticker_missing"]})')
    if gaps['missing_share'] > MAX_MISSING_SHARE:
        out.warn('labels shift by rows, so missing dates make the real horizon longer than h days')

    stats = signal_descriptives(df, signal_cols)
    warnings = signal_warnings(stats)
    for w in warnings:
        out.warn(w)
    if not warnings:
        out.ok('no signal is empty, constant or mostly null')

    # ------------------------------------------------------------------
    # [4/4] Save
    # ------------------------------------------------------------------
    out.step(4, 4, 'Save the signals and the provenance (git commit, input hash, parameters)')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / 'signals.parquet'
    df.write_parquet(out_file)

    config = {
        'run_id': run_id,
        'signal_tag': tag,
        'run_date': run_date,
        'run_timestamp': datetime.now().isoformat(),
        'signal_specs': SIGNAL_SPECS,
        'skipped_specs': skipped,
        'signal_cols': signal_cols,
        'cheat': cheat,
        'cheat_specs': cheat_specs,
        'cheat_cols': cheat_cols,
        'cheat_seed': CHEAT_SEED if cheat else None,
        'cheat_noise_days': CHEAT_NOISE_DAYS if cheat else None,
        'placebo': placebo,
        'placebo_cols': placebo_cols,
        'placebo_seed': PLACEBO_SEED if placebo else None,
        'placebo_noise_days': PLACEBO_NOISE_DAYS if placebo else None,
        'horizons': HORIZONS,
        'date_start': DATE_START,
        'date_end': DATE_END,
        'n_signals': len(signal_cols),
        'n_rows': df.height,
        'output_cols': output_cols,
        'source': cfg.source, 'asset': cfg.asset, 'region': cfg.region, 'universe': cfg.universe, 'freq': cfg.freq,
        # provenance: which code and which data made this file
        'git_commit': get_git_commit(),
        'git_diff': get_git_diff() or None,
        'features_file': str(cfg.features_file),
        'features_hash': file_hash(cfg.features_file),
        'params_hash': params_hash([SIGNAL_SPECS, HORIZONS, DATE_START, DATE_END, cheat_specs, CHEAT_SEED if cheat else None,
                                    CHEAT_NOISE_DAYS if cheat else None, PLACEBO_COUNT if placebo else None,
                                    PLACEBO_SEED if placebo else None, PLACEBO_NOISE_DAYS if placebo else None]),
        'script_path': str(Path(__file__).resolve()),
        # statistics that use NO returns
        'signal_stats': stats,
        'warnings': warnings,
        'label_checks': label_report,
        'calendar_gaps': gaps,
        'universe': universe_info,
        'n_dead_tickers': len(dead_set),
    }
    with open(out_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2, default=str)

    out.info('time', f'{time.time() - start:.1f} s')
    out.done({'signals': out_file, 'config': out_dir / 'config.json'},
             f'python scripts/pipeline/05_evaluate_signals.py --date {run_date} --tag {tag}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate the candidate signals and the forward returns')
    parser.add_argument('--date', type=str, required=True, help='Run date, YYYY-MM-DD (used in the run_id)')
    parser.add_argument('--tag', type=str, default='default', help='Run tag (default: default)')
    parser.add_argument('--cheat', action='store_true',
                        help='Add the cheating benchmark signals (CHEAT_SIGNALS). The tag must contain "cheat".')
    parser.add_argument('--placebo', action='store_true',
                        help='Add PLACEBO_COUNT signals of pure noise (true IC = 0). The tag must contain "placebo".')
    args = parser.parse_args()
    main(args.date, args.tag, args.cheat, args.placebo)
