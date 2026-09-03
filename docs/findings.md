# Research Findings

Dated, append-only. Each entry: what was tested, what was found, FDR status, robustness check. 
Cross-reference config/commit where useful for reproducibility.


## 2026-08-20: Momentum/reversal dead zone in `mom` across horizons

The momentum/reversal dead zone. Testing `mom` across horizons (5, 10, 20, 60, 120, 252 days) surfaces two distinct, literature-consistent effects rather than one smooth signal. 
    - Short horizons (5, 10, 20 days) show significant negative IC, short-term reversal, consistent with Jegadeesh (1990). 
    - The long horizon (252 days) shows significant positive IC, medium-term momentum, consistent with Jegadeesh & Titman (1993). 
    - Between them, roughly 60-120 days, IC is weak and doesn't survive BY correction. 
This is a real structural feature of the data, not a bug: reversal and momentum are documented as separate phenomena with different (in fact opposite-signed) economic mechanisms, a dead zone between them is exactly what the literature would predict, not an artifact to explain away. Confirmed independently via `long_short_sharpe` sign consistency across the two regimes, not just the IC sign alone.

## 2026-09-02: Isolated autocorrelation at lags ~15-16 and ~32 in market returns** 
Beyond the expected lag-1 reversal, lag-wise autocorrelation testing (FDR-corrected, `alpha=0.05`, BY) found two further isolated significant lags, disconnected from any contiguous decay from the origin. Not used for embargo sizing (see `methodology.md`), but worth investigating on their own, plausible candidates: biweekly/monthly rebalancing flows, options-expiration-adjacent effects, not yet checked against an actual calendar. Open question, no conclusion yet.

About the 1-lag autocorrelation: The leading explanation in the literature is nonsynchronous trading, sometimes called the Fisher effect after Lawrence Fisher's original 1966 observation. Not every stock in your panel trades right up to the close, some print their last trade a few minutes (or, for smaller/less liquid names, longer) before the actual close. When you average returns across many such stocks, today's cross-sectional average partly reflects information that some constituents hadn't fully incorporated yet, that gets included instead in tomorrow's average. This mechanically induces positive autocorrelation in a diversified portfolio's returns even when no individual stock shows meaningful predictability. Lo & MacKinlay's An Econometric Analysis of Nonsynchronous Trading (Journal of Econometrics, 1990) formalized this. A related, compounding effect: information genuinely diffuses across stocks at different speeds, large, heavily-covered names react to news faster than smaller ones, so a systematic move can partly "continue" the next day as slower-reacting names catch up.