# Formatting Rules
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
  01_download_data.py      # YFinance → data/raw/prices.parquet
  02_etl_prices.py         # Clean/align → data/processed/prices.parquet
  03_build_features.py     # Technical features → data/processed/features.parquet
  04a_generate_signals.py  # On-demand signals → data/processed/signals/{date}_{tag}/signals.parquet
  04b_evaluate_signals.py  # IC, FDR, decay → data/exports/signal_eval_{tag}/
  04c_fit_combiner.py      # Elastic Net → data/exports/combiner_{tag}/
  04d_estimate_risk_model.py  # RMT correlation + Louvain clustering
  04e_construct_weights.py # Portfolio optimization with risk model
  04f_run_backtest.py      # Full backtest with costs
  04_inspect.py            # ← DEPRECATED, use scripts/analysis/inspect_signals.py

scripts/analysis/     # Exploration tools (run anytime):
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
- **Signals are generated on-demand** (04a) — no persistent signals.parquet in processed/, versioned by `{date}_{tag}/`
- **Config is code** — `SIGNAL_SPECS` in 04a defines what signals to build; config.json saved per run for reproducibility
- **Forward returns use open-to-open** (`fwdret{h}` = logret from open(t+1) to open(t+h+1), realistic execution)
- **Evaluation horizon** = 20 days default (fwdret20)
- **Elastic Net combiner** — CPCV/Walk-forward CV, q-value weighting, FDR-filtered signals only
- **Risk model** — RMT eigenvalue cleaning + Louvain clustering on correlation matrix
- **Provenance** — Every run saves git_commit, features_hash, params_hash, signal_stats, ic_stats, performance_stats

## Common Commands
```bash
# Generate signals
python scripts/pipeline/04a_generate_signals.py

# Evaluate signals
python scripts/pipeline/04b_evaluate_signals.py --signal-dir data/processed/signals/2026-10-29_default

# Inspect runs
python scripts/analysis/inspect_signals.py list
python scripts/analysis/inspect_signals.py stats --run 2026-10-29_default
python scripts/analysis/inspect_signals.py ic --run 2026-10-29_default --horizon 20 --plot
python scripts/analysis/inspect_signals.py compare --runs 2026-10-29_default 2026-10-29_mom_rev

# Full pipeline
python scripts/pipeline/04_run_all.py
```

## Data Files
| Path | Description |
|------|-------------|
| `data/raw/prices.parquet` | Raw OHLCV from YFinance |
| `data/processed/prices.parquet` | Cleaned, aligned prices |
| `data/processed/features.parquet` | Technical features (mom, vol, beta, adv, etc.) |
| `data/processed/signals/{date}_{tag}/signals.parquet` | Generated signals + fwd returns |
| `data/processed/signals/{date}_{tag}/config.json` | Full run config + stats |
| `data/exports/signal_eval_{tag}/` | IC, FDR, decay reports |
| `data/exports/combiner_{tag}/` | Elastic Net coefficients, predictions |
| `data/exports/risk_model_{tag}/` | Correlation matrices, clusters |
| `data/exports/weights_{tag}.parquet` | Portfolio weights per date |

## Environment
- Python 3.11+ (uses `str|list` union syntax)
- Key deps: polars, numpy, scipy, sklearn, matplotlib, networkx
- Run `pip install -e .` in venv

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