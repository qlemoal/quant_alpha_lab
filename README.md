# Alpha Research Framework

A modular quantitative research framework built from scratch to reproduce the workflow used by systematic investment firms.

The objective of this project is **not** to produce a single profitable strategy, but to build a reusable research infrastructure capable of supporting many different alpha ideas.

The project emphasizes:

- reproducibility
- modularity
- computational efficiency
- statistical rigor
- realistic quantitative research practices

---

# Objectives

The long-term goal is to reproduce the complete research pipeline used by professional quantitative researchers.

The framework will support

- downloading raw market data
- data validation and cleaning
- feature engineering
- alpha generation
- statistical evaluation
- false discovery control (FDR)
- portfolio construction
- backtesting
- robustness testing
- signal comparison

Every new alpha should require only writing the signal itself. Everything else should already exist.

---

# Technologies

## SQL + DuckDB

Used for operations involving millions of observations:

- imports
- joins
- filtering
- aggregations

DuckDB operates directly on parquet files and avoids loading unnecessary data into memory.

---

## Polars

Used for feature engineering.

Reasons:

- lazy evaluation
- automatic parallelization
- very fast rolling operations
- memory efficient

The goal is to keep every transformation lazy until the final collect().

---

## Pandas

Only used after reducing the dataset to a manageable size.

Main uses:

- plotting
- debugging
- exploratory analysis

---

## Matplotlib

Used for every visual sanity check.

Every transformation should be inspected before moving to the next stage.

---

# Repository structure

```
config/
    Project configuration

data/
    raw/
        Original downloaded data

    processed/
        Clean datasets ready for research

    intermediate/
        Temporary files

    exports/
        Figures and reports

scripts/
    Executable scripts

signals/
    One file = one alpha idea

features/
    Feature engineering

portfolio/
    Portfolio construction

evaluation/
    IC, Sharpe, robustness, FDR

utils/
    Reusable helper functions

sql/
    SQL queries

tests/
    Unit tests

notebooks/
    Exploration and debugging
```

---

# Development philosophy

This project intentionally avoids automatic code generation.

The objective is to understand every design decision and reproduce professional research workflows rather than maximizing coding speed.

Large language models are used as:

- technical guidance
- code review
- debugging assistance
- discussion of best practices

All implementation decisions remain manual and are understood before being added to the project.

---

# Workflow

```
Download

↓

Clean

↓

Inspect prices

↓

Build features

↓

Inspect features

↓

Generate alpha

↓

Evaluate signal

↓

Control false discoveries

↓

Portfolio construction

↓

Backtest

↓

Performance attribution
```

---

# Notebooks

Notebooks are used only for

- exploration
- plotting
- debugging
- hypothesis generation

Reusable code should never stay inside notebooks.

---

# Helper imports

Every notebook starts by importing `setup.py`.

The setup script

- adds the project root to `sys.path`
- imports the most common libraries
- sets plotting defaults
- imports commonly used helper functions

This keeps notebooks concise while allowing reusable code to live outside the notebook itself.

---

# Current Progress

- ✔ Data download
- ✔ Data cleaning
- ✔ Price inspection
- ⬜ Feature engineering
- ⬜ Alpha library
- ⬜ Portfolio construction
- ⬜ FDR framework
- ⬜ Random Forest models
- ⬜ Cross-validation