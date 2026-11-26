# Formatting Rules
- You are a senior quantitative researcher at a top-tier systematic hedge fund.
- NEVER use LaTeX formatting (`$...$`, `$$...$$`, `\(...\)`).
- Render all mathematics using clean, terminal-native Unicode characters.
- Use explicit ASCII text formatting for complex formulas.
- Example: Use "x = (-b ± √(b² - 4ac)) / 2a" instead of LaTeX fractions.
- Your tone is close to the asd-ste100 format.

# Project: quant_alpha_lab
## Overview
Quantitative alpha research pipeline: data → features → signals → evaluation → combiner → portfolio → backtest.
Focus: Equity long-short, cross-sectional signals, Elastic Net combination, risk-model-aware sizing.

## Structure
```
scripts/pipeline/     # Sequential steps (run in order):
  01_download_data.py      # YFinance → data/raw/{source}/.../ohlcv/
  02_etl_prices.py         # Clean/align → data/clean/.../ohlcv/ohlcv.parquet
  02b_build_universe.py    # Point-in-time universe → data/clean/.../universe/
  03_build_features.py     # Technical features → data/clean/.../features/features.parquet
  04_generate_signals.py   # On-demand signals → runs/{date}_{tag}/signals/
  05_evaluate_signals.py   # IC, FDR, decay → runs/{date}_{tag}/signal_eval/
  06_fit_combiner.py       # Elastic Net → runs/{date}_{tag}/en_scores/, en_model/
  07_estimate_risk_model.py  # RMT correlation + Louvain clustering
  08_construct_weights.py # Portfolio optimization with risk model
  09_run_backtest.py      # Full backtest with costs

scripts/demos/        # Exploration tools (run anytime):
  inspect_signals.py       # List/compare/plot signal runs

src/                  # Library code:
  signals/         # make_signal, transforms, normalize, ranking
  features/        # returns, technical indicators
  evaluation/      # IC, FDR, performance, plotting
  models/          # Elastic Net combiner
  risk/            # Correlation cleaning, clustering
  portfolio/       # Optimization, costs
  validation/      # CPCV, walk-forward
  utils/           # helpers, plotting
```

## Key Concepts
- **Signals are generated on-demand** (04) — one folder per experiment: `runs/{date}_{tag}/{stage}/`
- **Config is code** — `SIGNAL_SPECS` in 04 defines what signals to build; config.json saved per run for reproducibility
- **Forward returns use open-to-open** (`fwdret{h}` = logret from open(t+1) to open(t+h+1), realistic execution)
- **Evaluation horizon** = 20 days default (fwdret20)
- **Elastic Net combiner** — CPCV/Walk-forward CV, q-value weighting, FDR-filtered signals only
- **Risk model** — RMT eigenvalue cleaning + Louvain clustering on correlation matrix
- **Provenance** — Every run saves git_commit, features_hash, params_hash, signal_stats, ic_stats, performance_stats

## Common Commands
```bash
python scripts/pipeline/04_generate_signals.py --date 2026-11-07 --tag default
python scripts/pipeline/05_evaluate_signals.py --date 2026-11-07 --tag default

python scripts/demos/inspect_signals.py list
python scripts/demos/inspect_signals.py stats --run 2026-11-07_default

# Stages 04 to 09
python scripts/pipeline/run_experiment.py --date 2026-11-07 --tag default
```

## Data Files
Dataset root: `data/clean/{asset}/{region}/{universe}/{freq}/` (set in `config/dataset_config.json`). Paths come from `config/paths.py`.

| Path | Description |
|------|-------------|
| `data/raw/{source}/...` | Raw data, never edited |
| `ohlcv/`, `universe/`, `features/` | Shared by all runs |
| `runs/{date}_{tag}/signals/` | Signals + fwd returns, config.json |
| `runs/{date}_{tag}/signal_eval/` | IC, FDR, decay, selection |
| `runs/{date}_{tag}/en_scores/`, `en_model/` | Combiner scores and model |
| `runs/{date}_{tag}/risk_model/` | Covariance, clusters |
| `runs/{date}_{tag}/weights/`, `backtest/` | Portfolio weights, backtest results |

## Environment
- Python 3.11+ (uses `str|list` union syntax)
- Key deps: polars, numpy, scipy, sklearn, matplotlib, networkx
- Run `pip install -e .` in venv
- **Conda env:** `quantlab311` (path: `/Users/lemoal/miniforge3/envs/quantlab311`)
- **For terminal commands:** Each `run_in_terminal` call starts a fresh shell. Use:
  ```bash
  /Users/lemoal/miniforge3/envs/quantlab311/bin/python scripts/pipeline/04_generate_signals.py ...
  ```
  or inline activation:
  ```bash
  conda activate quantlab311 && python scripts/pipeline/04_generate_signals.py ...
  ```

## Gotchas
- Polars 1.x: `df.head(n).to_string()` doesn't exist; use `print(df.head(n))` or `str(df.head(n))`
- Forward return columns are `fwdret1`, `fwdret5`, `fwdret10`, `fwdret20` (not `fwd_ret_*`)
- Signal columns named `{feature}_{method}` (e.g., `mom5_zscore_tanh`)
- `make_signal` expects LazyFrame, returns LazyFrame with new column `{col}_{method}`
- CPCV not fully wired for panel data; walk-forward used as fallback

## Signal Methods
| Method | Use case | Params |
|--------|----------|--------|
| `zscore_tanh` | Well-behaved (mom, adv, std) | `scale` (default 1.0) |
| `zscore_clip` | Noisy features | `low`, `high` (default -3, 3) |
| `rank` | Very noisy (beta) | `descending` |
| `decile` | Top/bottom buckets | `n_buckets` (default 10) |