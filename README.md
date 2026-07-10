# Systematic Equity Research Platform

This project is a modular quantitative research platform for developing, testing and evaluating systematic equity trading signals.

The goal is not to implement a single trading strategy, but to build a reusable research framework similar to those used in systematic hedge funds.

## Objectives

The project aims to reproduce the complete quantitative research workflow:

- acquisition of historical market data
- data validation and cleaning
- feature engineering
- alpha signal research
- portfolio construction
- backtesting and evaluation
- statistical validation and false discovery control

The emphasis is placed on writing clean, modular and reproducible code rather than producing isolated notebooks.

---

## Technology stack

Each tool is used where it is most appropriate.

### DuckDB / SQL

Used for

- importing large datasets
- joins
- filtering
- aggregations
- data validation

SQL remains one of the most widely used languages in quantitative finance, and many firms rely heavily on database queries for research pipelines.

### Polars

Used for feature engineering.

Rolling statistics, lagged variables, ranking and group operations are implemented using Polars LazyFrames.

Reasons for choosing Polars:

- significantly faster than pandas
- lazy execution
- parallel execution
- excellent memory efficiency

### Pandas + Matplotlib

Only used for

- visualization
- debugging
- exploratory inspection

Large datasets remain in Polars until a small subset needs to be visualized.

---

## Repository structure

scripts/
    executable scripts

features/
    reusable feature engineering

signals/
    alpha signal generation

portfolio/
    portfolio construction

evaluation/
    performance evaluation

utils/
    reusable helper functions

notebooks/
    exploratory analysis and reports

data/

raw/
original downloaded datasets

processed/
clean datasets used throughout research

exports/
figures and reports

---

## Philosophy

The project follows several design principles.

- immutable raw data
- reproducible pipelines
- modular research
- no duplicated computations
- separation between features, signals and portfolios
- reproducible experiments

The objective is that adding a new alpha requires writing only a single new research module.

---

## AI assistance

The code is written manually.

Large language models are used only as technical advisors to discuss software architecture, quantitative research methodology, and implementation choices. Code is understood, adapted and tested before being integrated into the project.

No autonomous coding agents (e.g. Claude Code, Cursor Agent) are used.