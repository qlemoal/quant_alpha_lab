# Research Findings

Dated, append-only. Each entry: what was tested, what was found, FDR status, robustness check. 
Cross-reference config/commit where useful for reproducibility.


## 2026-08-20: Momentum/reversal dead zone in `mom` across horizons

The momentum/reversal dead zone. Testing `mom` across horizons (5, 10, 20, 60, 120, 252 days) surfaces two distinct, literature-consistent effects rather than one smooth signal. 
    - Short horizons (5, 10, 20 days) show significant negative IC, short-term reversal, consistent with Jegadeesh (1990). 
    - The long horizon (252 days) shows significant positive IC, medium-term momentum, consistent with Jegadeesh & Titman (1993). 
    - Between them, roughly 60-120 days, IC is weak and doesn't survive BY correction. 
This is a real structural feature of the data, not a bug: reversal and momentum are documented as separate phenomena with different (in fact opposite-signed) economic mechanisms, a dead zone between them is exactly what the literature would predict, not an artifact to explain away. Confirmed independently via `long_short_sharpe` sign consistency across the two regimes, not just the IC sign alone.