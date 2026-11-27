# The pipeline, step by step

Each step is one card: **Problem → Solution → Why → How (where in the code)**.
Read the diagrams first. The text is only there to explain them.

---

## The whole pipeline

```
 RAW DATA            CLEAN DATA                 IDEAS                  DECISION              MONEY
 ─────────           ──────────                 ─────                  ────────              ─────
 01 download  ──>  02 clean prices  ──>  03 features  ──>  04 signals  ──>  05 SELECT  ──>  06 COMBINE
                   02b universe                              + labels        "real or luck?"   "one score"
                                                                                                  │
            09 BACKTEST  <──  08 WEIGHTS  <──  07 RISK MODEL  <─────────────────────────────────────┘
            "money after       "how much        "how stocks
             costs, on the      of each          move together"
             holdout?"          stock?"
                │
                v
         KEEP  or  DROP
```

| Stages | Question they answer |
|---|---|
| 01 · 02 · 02b | Do we have correct prices, for the stocks we could really trade? |
| 03 · 04 | What could predict returns? |
| 05 · 06 | Does it really predict returns, or is it luck? |
| 07 · 08 | How do we turn a prediction into a safe book? |
| 09 | Would it have made money, after costs, on data nobody looked at? |

---

## Rule 0: the clock

```
 day t                          day t+1                              day t+21
 ──────────────────────────────────────────────────────────────────────────────>
   close of t                     open of t+1                        open of t+21
   last data we use               we BUY / SELL here                 the label ends here
   signal is computed             (earliest possible trade)
   │                              │<──────────── label = 20-day return ────>│
```

| | |
|---|---|
| **Problem** | Using a price you could not have known yet (look-ahead) makes every backtest look great. |
| **Solution** | Compute the signal at the close of t, trade at the open of t+1, and measure the label from open to open. |
| **Why** | It is the earliest trade that is really possible. Nothing in the future can leak in. |
| **How** | `fwdret20(t) = log open(t+21) − log open(t+1)` in `src/features/returns.py` |

---

## 01 · Download the raw data

```
 Yahoo / CRSP / vendor  ──>  data/raw/{source}/...   (never edited again)
```

| | |
|---|---|
| **Problem** | Data vendors change their data afterwards. If you edit files by hand, a result can no longer be reproduced. |
| **Solution** | Store the raw files once and never touch them. Every later stage reads them and writes new files. |
| **Why** | Any result can then be rebuilt from the same raw data. |
| **How** | `01_download_data.py` (Yahoo), `01_import_crsp.py` (with delisted stocks), `01_import_vendor_data.py` |

---

## 02 · Clean the prices

```
 raw CSV ──> read as text ──> drop bad rows ──> 1 row per (stock, day) ──> adjust the open ──> ohlcv.parquet
                               (count each reason)                        open × close / close_raw
```

| | |
|---|---|
| **Problem** | Bad rows, duplicates, and an open that is not adjusted for splits and dividends. An unadjusted open shows a fake −50% on a 2-for-1 split. |
| **Solution** | Drop bad rows and count why. Keep one row per day. Put the open on the same adjusted scale as the close. |
| **Why** | We trade at the open, so the open must be right. |
| **How** | `02_etl_prices.py` |

---

## 02b · The universe: only stocks you could trade THAT day

```
                  2015        2018        2021        2024
 stock A (alive)  ████████████████████████████████████████
 stock B (died)   ██████████████████████████▌                  <- disappears from today's index list
 stock C (new)                      ██████████████████████████

 today's list      : A, C           -> B is missing: only winners -> backtest too good
 point-in-time     : 2015 {A, B}, 2021 {A, B, C}, 2024 {A, C}  -> what you could trade then
```

| | |
|---|---|
| **Problem** | Survivorship bias. A list of today's stocks contains only the companies that survived. The ones that went bankrupt are missing. |
| **Solution** | On each date, use only the stocks that existed and were liquid on that date: the top N by dollar volume, re-ranked every k days. |
| **Why** | A backtest on survivors is biased upwards, sometimes by several % a year. |
| **How** | `02b_build_universe.py`. Its audit says FAIL when the file has no dead stocks (survivors only). Details in `docs/survivorship.md` |

---

## 03 · Features: raw measurements of each stock

```
 prices ──┬──> momentum    mom5 ... mom252     (return over the last w days)
          ├──> volatility  vol5 ... vol252     (std of daily returns)
          ├──> liquidity   log_adv10 ... 60    (log of dollars traded per day)
          └──> beta        beta60, beta252     (sensitivity to the market)
                  ^ every window looks BACKWARD only
```

| | |
|---|---|
| **Problem** | A feature that uses even one future day is look-ahead. |
| **Solution** | Use trailing windows only, computed for each stock separately. |
| **Why** | Short windows capture reversal, long windows capture trend. The data decides which ones work, in stage 05. |
| **How** | `03_build_features.py`. A test cuts the prices at a date and checks that no earlier feature changes. |

---

## 04 · Signals: compare the stocks of the same day

```
 day t:   stock     mom20      z-score across stocks   tanh       signal
          AAPL      +8%    ->        +1.4          ->  +0.89
          XOM       −2%    ->        −0.6          ->  −0.54
          ...
          (the market level disappears: only "better or worse than the others today" is left)
```

| | |
|---|---|
| **Problem** | Raw features mix two things: "is this stock better than the others?" and "is the market up?". The book is long-short, so only the first one matters. |
| **Solution** | Transform each feature across the stocks of the same date (z-score, then tanh or clip, or rank). |
| **Why** | It removes the market level and squeezes outliers, so one extreme stock cannot drive the signal. |
| **How** | `04_generate_signals.py`, `SIGNAL_SPECS`. Stage 04 never looks at returns, so the choice of signals cannot be tuned on results. |

### 04b · Benchmarks: test the machine itself

```
 PLACEBO  = pure noise              true IC = 0       ──>  must NEVER be selected
 CHEAT    = peeks at the future     true IC = known   ──>  must be found when strong enough
```

| | |
|---|---|
| **Problem** | On real data, you never know the truth. A broken test can look fine. |
| **Solution** | Add fake signals whose true strength is known, and check that the pipeline finds exactly that. |
| **Why** | It is the only direct proof that the selection is neither blind nor too easy to fool. |
| **How** | `run_experiment.py --cheat --placebo` (the tag must contain `cheat` and `placebo`) |

---

## The time split: decide on one part, judge on another

```
 2016 ──────────────────────────────── 2022-07 ░░░ 2022-09 ─────────────────── 2025-08
 │<──────────── SEARCH ───────────────────>│ gap │<──────── HOLDOUT ─────────>│
   05 selects, 06 fits and tunes            37 d   locked until stage 09
                                                   756 days ≈ 3 years
```

| | |
|---|---|
| **Problem** | If you choose and judge on the same data, the judgement is biased: you chose what worked there. |
| **Solution** | Choose on the search period only. Judge once, at the end, on the last 3 years. |
| **Why the gap** | A label is a 20-day forward return. Without a gap, the last search labels would contain holdout prices. |
| **Why 3 years** | The 95% range of a Sharpe ratio is about ±1.96 / √years: ±1.1 with 3 years, ±2.0 with 1 year. One year proves nothing. |
| **How** | `HOLDOUT_DATES = 756` in `config/constants.py` |

---

## 05 · Select: real signal or luck?

### 05a · Measure: one IC per day

```
 each day:  rank of the signal  vs  rank of the next 20-day return   (300 stocks)
                          └─── correlation ───┘
                                   = IC(day)        0 = useless,  +0.03 = good

 search period:  IC(1), IC(2), ..., IC(1702)   ──>  mean IC
```

| | |
|---|---|
| **Problem** | You need one number for "does the signal rank the stocks correctly?". |
| **Solution** | The daily rank correlation (IC) between the signal and the next return. |
| **Why** | Ranks ignore outliers, and the order of the stocks is exactly what a long-short book uses. |

### 05b · Three gates, in order

```
 signal ──> [1 PRIOR?] ──> [2 SIGN OK?] ──> [3 NOT LUCK?] ──> KEPT
               │ no           │ no             │ no
               v              v                v
            dropped        dropped          dropped
```

| Gate | Problem | Solution | Why |
|---|---|---|---|
| 1 prior | With many tries, some pattern always looks good. | An economic story, written BEFORE the test (`src/signals/priors.py`). | Fewer free choices means less data mining. |
| 2 sign | Flipping a signal after you see the result is snooping. | The measured IC must have the sign the story predicts. | The effect must be the one we believe in. |
| 3 not luck | The IC series has memory, and we test many signals. | Block bootstrap, then Romano-Wolf, then earlier trials (below). | Keeps the false-positive rate at a known 5%. |

### 05c · Gate 3, part 1: the data has memory, so resample BLOCKS

```
 neighbouring days share 19 of their 20 return days:
   day t    |====================|
   day t+1   |====================|
   -> 1702 ICs hold only about 30 independent pieces of information

 plain t-test:    assumes 1702 independent days -> se about 7× too small -> fake signals

 block bootstrap: real  [ A A A | B B B | C C C | D D D ]
                  fake  [ C C C | A A A | D D D | C C C ]   <- the memory inside blocks is kept
                  repeat 1000×  ->  how big a t you get by luck alone
```

| | |
|---|---|
| **Problem** | A t-test that ignores memory finds signals that are not there. Newey-West also says "significant" 9 to 22% of the time when it should be 5%. |
| **Solution** | Stationary block bootstrap: blocks of random length (mean 58 days here, never below 20), with the noise (se) re-estimated in every resample. |
| **Why** | Measured on simulations: its false-positive rate stays at 4 to 5%, even for slow signals. |
| **How** | `src/evaluation/signals/bootstrap.py`. Proof: `scripts/research/calibrate_tests.py` |

```
 says "significant" on pure noise (should be 5%):
                     signal memory 1 day   20 days   60 days
   Newey-West               8.7%            9.7%      22.0%
   bootstrap (used)         3.7%            4.3%       4.3%
```

### 05d · Gate 3, part 2: many signals tested at once

```
 15 signals tested at 5% each  ->  about 1 pure-noise signal passes by luck

 Romano-Wolf:  "is the BEST of 15 noise signals ever this good?"
               uses the same resampled days for all signals -> keeps their real correlation

 example (reference run):   mom120   p alone 0.023  ->  p among 15: 0.100  ->  dropped
```

| | |
|---|---|
| **Problem** | The more signals you test, the more lucky ones pass. |
| **Solution** | Romano-Wolf step-down: the chance that ANY kept signal is luck stays below 5%. |
| **Why not Bonferroni** | Our signals are correlated (mom60 and mom120 move together). Romano-Wolf uses that, so it is less strict for no reason. |

### 05e · Gate 3, part 3: the runs you made before

```
 run 1: 15 signals ─┐
 run 2: 15 signals ─┼──>  45 tries in total, not 15
 run 3: 15 signals ─┘

 p_all = min(1, p_RW × (K + n_trials_extra) / K)        e.g.  0.01 × 45/15 = 0.03
```

| | |
|---|---|
| **Problem** | "I changed things until it worked." Romano-Wolf sees only the current run. |
| **Solution** | Multiply by the number of runs made on the same data. You set `--n_trials_extra` yourself. |
| **Why** | This is the main defence against overfitting by trial and error. |

**Cross-checks** (Newey-West, FDR q-values, deflated Sharpe, IC by year, IC decay) are printed but decide nothing. A big disagreement with the gates is a warning sign.

---

## 06 · Combine: one score from the kept signals

```
 kept signals ──> × prior sign ──> average inside each family ──> Elastic Net ──> one score
                                   momentum: mom5, mom252          weights ≥ 0
                                   liquidity: adv10, 20, 60        one-SE rule
```

| | |
|---|---|
| **Problem** | Kept signals overlap (adv10, adv20 and adv60 measure almost the same thing). A free regression on them overfits and can flip signs. |
| **Solution** | Average inside each family first, then use an Elastic Net with non-negative weights, and take the most cautious fit within one standard error of the best one. |
| **Why** | Few parameters, signs that keep their economic meaning, and a bias toward simplicity. |
| **How** | `06_fit_combiner.py`, `src/models/elastic_net_combiner.py`. Details in `docs/selection.md` §3 |

### 06b · Tuning without leaking: purged CPCV

```
 search period cut into 10 blocks, 2 are the test each time (45 combinations):

   blocks:  1   2   3   4   5   6   7   8   9   10
   fold:    .   T   .   .   .   T   .   .   .   .        T = test   . = train
               ▓▓T▓▓           ▓▓T▓▓
               ^   ^
               purge before (labels overlap the test), embargo after
```

| | |
|---|---|
| **Problem** | A random K-fold puts day t in training and day t+5 in test. They share 15 days of return, so the answer leaks into the training data. |
| **Solution** | Split by blocks of time. Remove the days around each test block whose labels overlap it (purge and embargo). |
| **Why** | Each search day then gets a score from models that never saw it or its neighbours (an out-of-fold score). |
| **Limit** | Those models trained on the days AFTER it too, so the search-period score is not tradable live. Only the holdout is. |

---

## 07 · Risk model: how the stocks move together

```
 252 days × 300 stocks ──> correlation ──> remove the noise ──> shrink ──> covariance Σ
                                           (Marchenko-Pastur)  (Ledoit-Wolf)
                                   │
                                   └──> remove the market mode ──> Louvain ──> clusters
                                                                             (industry-like groups)
```

| | |
|---|---|
| **Problem** | 300 stocks and 252 days give a correlation matrix that is mostly noise. An optimiser would bet on that noise. |
| **Solution** | Keep only the eigenvalues above the pure-noise limit (random matrix theory), then shrink toward simple. Find groups of stocks that move together from the data. |
| **Why** | A stable Σ to size the book, and groups so that the book does not become one sector bet. |
| **How** | `07_estimate_risk_model.py`, `src/risk/`. Re-estimated on each weekly rebalance date, from past data only. |

---

## 08 · Weights: from a score to a safe book

```
 score ──> 1 weights ∝ score ──> 2 cap each cluster ──> 3 long $ = short $ ──> 3b beta = 0
                                                                                   │
              6 checks <── 5 gross ≤ 4× capital <── 4 scale to 10% volatility <─────┘
```

| Step | Problem | Solution |
|---|---|---|
| 1 | Turn a score into positions. | Weight proportional to the demeaned score: simple and transparent. |
| 2 | Most of the risk could sit in one industry. | Cap the gross exposure of each cluster. |
| 3 | Capping breaks the balance. | Long dollars = short dollars again. |
| 3b | Equal dollars can still be long the market (long high-beta stocks, short low-beta). | The smallest change that makes the beta 0. |
| 4 | The same book is riskier in a storm than in a calm market. | Scale the book to 10% expected volatility. |
| 5 | Unlimited borrowing is not realistic. | Gross exposure at most 4 × capital. |
| 6 | A silent bug. | Stop if the book is not neutral or not finite. |

**How:** `08_construct_weights.py`

---

## 09 · Backtest: money after costs, judged on the holdout

```
 weights at close t ──> trade at open t+1 ──> hold (weights drift) ──> next rebalance: trade the difference
                         │
                         └── pay: half-spread 3 bps + commission 1 bp + impact ∝ √(trade / daily volume)

 dead stock: the position is closed at its last price, the day after its last row
```

| | |
|---|---|
| **Problem** | A signal can predict returns and still lose money: each trade costs something. |
| **Solution** | Simulate each day with realistic trading, drift and costs. Judge on the holdout only. |
| **Why drift** | Rebalancing to the target every day would hide most of the turnover. |
| **Why dead stocks** | A book that is long a stock that went bankrupt must take the loss. |
| **How** | `09_run_backtest.py`, `src/portfolio/weights_backtest.py`, `src/portfolio/costs.py` |

### 09b · How to read the result

```
                      before costs    trading costs    after costs
 return per year          +42%     −      39%       =      +4%     <- where the money goes

 Sharpe after costs  +0.44      95% range  [−0.7 ─────●───── +1.6]
                                                 0 is inside -> could be luck
```

| Check | Pass when | Why |
|---|---|---|
| Skill after costs | the 95% range of the holdout Sharpe is above 0 | anything else could be luck |
| Survives higher costs | Sharpe > 0 at 2× the cost assumption | the cost model is a guess |
| Market neutral | beta between −0.1 and +0.1 | otherwise it is market exposure, not skill |
| Risk model right | realised / planned volatility between 0.7 and 1.3 | otherwise the sizing is wrong |
| Steady | most years gain after costs | one lucky year can carry a whole sample |

---

## What this pipeline does NOT protect against

| Limit | Effect |
|---|---|
| Yahoo data = survivors only | Results are biased upwards. Use CRSP (with dead stocks) to be sure. |
| No borrow fee or financing cost | Results are a little too good. |
| Only one holdout path of 3 years | A Sharpe is still known only to about ±1.1. |
| Ideas never tested are not counted | `n_trials_extra` counts only what you report. |
| The test assumes the past is one stable world | A signal that died in 2021 is averaged with its good years. Read the IC by year. |
