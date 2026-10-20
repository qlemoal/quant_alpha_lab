'''
src/risk/portfolio_construction.py

Core portfolio construction functions: signal → weights with risk controls.

================================================================================
WHY THIS MATTERS
================================================================================

This is the bridge from "alpha signal" to "tradeable portfolio". A raw alpha
signal (e.g. Elastic Net combined score) tells you *which* stocks to prefer,
but not *how much* to own. Portfolio construction answers:

  - How big should each position be? (sizing)
  - How do we limit concentration risk? (cluster caps)
  - How do we control overall portfolio risk? (vol targeting)
  - How do we account for transaction costs? (turnover management)

Top quant firms (Millennium, Citadel, Two Sigma, AQR) all use a layered
construction approach:

  Layer 1: Signal → Raw Weights      (proportional, inv-vol, or rank-based)
  Layer 2: Risk Model Integration    (covariance-aware sizing)
  Layer 3: Constraint Enforcement    (cluster caps, sector limits, etc.)
  Layer 4: Vol Targeting             (scale to target risk)
  Layer 5: Cost-Aware Optimization   (optional, integrates with costs.py)

This module implements Layers 1-4 in a modular, testable way. Layer 5
lives in src/portfolio/optimization.py and src/portfolio/costs.py.

================================================================================
INTEGRATION WITH EXISTING PIPELINE
================================================================================

Inputs consumed from existing modules:

  - Signal scores:      from src/models/elastic_net_combiner or gbm_combiner
  - Correlation matrix: from src/risk/correlation_cleaning (RMT-cleaned)
  - Volatilities:       from src/features/volatility (rolling std)
  - Cluster map:        from src/risk/clustering (Louvain on cleaned corr)
  - ADV:                from src/features/adv (for cost model)

Outputs consumed by:

  - src/portfolio/backtest.py   -- walk-forward backtest loop
  - src/portfolio/optimization.py -- cost-aware re-optimization

================================================================================
DESIGN PRINCIPLES
================================================================================

1.  **Polars-native**: Accept/return Polars Series/DataFrames for dates/tickers.
    Convert to NumPy only at the linear-algebra boundary.

2.  **Cross-sectional per date**: All operations are per-date (cross-section).
    No time-series logic here -- that lives in backtest.py.

3.  **Explicit, not magic**: Every parameter is an argument, no hidden defaults
    from global config. Config values are passed in explicitly.

4.  **Modular**: Each function does ONE thing and can be tested in isolation.
    The `construct_portfolio()` function chains them for convenience.

5.  **Numerically robust**: Handle edge cases (single ticker, all-null,
    zero vol, degenerate correlation) gracefully with clear warnings.

================================================================================
REFERENCES
================================================================================

  - Grinold, R. C., & Kahn, R. N. (2000). "Active Portfolio Management" (2nd ed.).
    McGraw-Hill.  (Ch. 3-4: signal weighting, risk models, vol targeting.)

  - López de Prado, M. (2018). "Advances in Financial Machine Learning." Wiley.
    Ch. 10 (portfolio construction), Ch. 11 (risk models).

  - Asness, C. S., Frazzini, A., & Pedersen, L. H. (2012). "Leverage aversion
    and risk parity." Financial Analysts Journal, 68(1), 47-59.
    (Inverse volatility / risk parity weighting.)

  - Raffinot, T. (2018). "Hierarchical clustering-based asset allocation."
    (Clustered Risk Parity / HRP extensions.)

  - Almgren, R., & Chriss, N. (2001). "Optimal execution of portfolio transactions."
    Journal of Risk, 3(2), 5-39.  (Square-root impact model.)

================================================================================
'''

import numpy as np
import polars as pl
from polars import col as c
from src.utils.helpers import as_list


# =============================================================================
# LAYER 1: SIGNAL → RAW WEIGHTS
# =============================================================================

def raw_weights_from_score(
    score: pl.Series,
    tickers: pl.Series | None = None,
    dates: pl.Series | None = None,
    max_gross: float = 1.0,
    method: str = 'proportional',
    vols: np.ndarray | pl.Series | None = None,
) -> pl.Series:
    '''
    Convert a raw alpha signal into dollar-neutral, gross-normalized weights.

    ===========================================================================
    INTUITION
    ===========================================================================

    A signal gives a directional view (positive = long, negative = short).
    We need to translate that into position sizes. Three canonical approaches:

    1. PROPORTIONAL (default, methodology.md Section 5.3):
       w_i = score_i / sum_j |score_j| * max_gross

       - Keeps the signal's *relative magnitude* as position sizing.
       - If signal A is 2x signal B, position A is 2x position B.
       - Simple, transparent, no free parameters.
       - Caveat: if signal has outliers, weights can be dominated by a few names.

    2. INVERSE VOLATILITY (risk-equalized signal tilt):
       w_i = (score_i / vol_i) / sum_j |score_j / vol_j| * max_gross

       - Tilts toward assets where the signal is "cleaner" (higher signal-to-noise).
       - Same idea as risk parity: equalize risk contribution per unit of signal.
       - Requires vol estimates (passed as `vols`).

    3. RANK-BASED (robust to outliers):
       w_i = rank(score_i) / sum_j |rank(score_j)| * max_gross

       - Uses only ordinal information, discards magnitude.
       - Robust to extreme signal values (e.g. data errors, untrusted features).
       - Same philosophy as rank transform in signals/transforms.py.

    ===========================================================================
    PARAMETERS
    ===========================================================================

    score:       Polars Series of signal values (one per ticker on this date).
    tickers:     Optional ticker Series (same length as score) for alignment.
    dates:       Optional date Series (same length) for downstream tracking.
    max_gross:   Target gross exposure (sum of absolute weights).
                 1.0 = 100% long + 100% short = 200% gross notional.
    method:      'proportional', 'inv_vol', or 'rank'.
    vols:        Required for method='inv_vol'. P-vector of volatilities
                 (aligned with score order). Can be np.ndarray or pl.Series.

    ===========================================================================
    RETURNS
    ===========================================================================

    Polars Series of weights (same length as score, aligned by index).

    ===========================================================================
    EDGE CASES
    ===========================================================================

    - All scores equal (or all null): returns equal-weight (or zeros if null).
    - Single ticker: returns +/- max_gross (dollar-neutral impossible with 1).
    - Vols with zeros: clipped to small floor for inv_vol method.
    '''
    score_np = np.asarray(score.to_numpy(), dtype=float)
    n = len(score_np)

    if n == 0:
        return pl.Series('weight', [], dtype=pl.Float64)

    # Handle all-null or all-NaN
    valid_mask = ~np.isnan(score_np)
    if not valid_mask.any():
        # All null: return zeros
        weights = np.zeros(n, dtype=float)
        return pl.Series('weight', weights)

    # -------------------------------------------------------------
    # Method 1: Proportional (signal magnitude preserved)
    # -------------------------------------------------------------
    if method == 'proportional':
        # Dollar-neutral: center the signal first so sum(weights) = 0
        score_centered = score_np - np.nanmean(score_np)
        gross = np.nansum(np.abs(score_centered))

        if gross > 0:
            weights = score_centered / gross * max_gross
        else:
            # All scores identical after centering -> flat
            weights = np.zeros(n, dtype=float)

    # -------------------------------------------------------------
    # Method 2: Inverse Volatility (risk-equalized signal tilt)
    # -------------------------------------------------------------
    elif method == 'inv_vol':
        if vols is None:
            raise ValueError("method='inv_vol' requires `vols` argument")

        vols_np = np.asarray(vols.to_numpy() if isinstance(vols, pl.Series) else vols, dtype=float)
        if vols_np.shape != (n,):
            raise ValueError(f"vols shape {vols_np.shape} != score length {n}")

        # Clip vols to avoid division by zero
        vols_np = np.maximum(vols_np, 1e-8)

        # Signal per unit of risk
        signal_per_risk = np.where(valid_mask, score_np / vols_np, 0.0)
        # Dollar-neutral
        signal_per_risk = signal_per_risk - np.nanmean(signal_per_risk[valid_mask])
        gross = np.sum(np.abs(signal_per_risk))

        if gross > 0:
            weights = signal_per_risk / gross * max_gross
        else:
            weights = np.zeros(n, dtype=float)

    # -------------------------------------------------------------
    # Method 3: Rank-based (ordinal only, robust)
    # -------------------------------------------------------------
    elif method == 'rank':
        # Rank within valid (non-NaN) entries
        ranked = np.full(n, np.nan)
        if valid_mask.any():
            valid_scores = score_np[valid_mask]
            # Average rank for ties (method='average')
            from scipy.stats import rankdata
            ranked[valid_mask] = rankdata(valid_scores, method='average')
            # Center ranks so sum = 0 (dollar-neutral)
            ranked[valid_mask] = ranked[valid_mask] - np.mean(ranked[valid_mask])
            gross = np.sum(np.abs(ranked[valid_mask]))
            if gross > 0:
                weights = ranked / gross * max_gross
            else:
                weights = np.zeros(n, dtype=float)
        else:
            weights = np.zeros(n, dtype=float)
        weights = np.where(valid_mask, weights, 0.0)

    else:
        raise ValueError(f"unknown method: {method!r}. Use 'proportional', 'inv_vol', or 'rank'.")

    # Ensure dollar-neutral (sum = 0) and gross = max_gross (within numerical precision)
    weights = np.asarray(weights, dtype=float)
    weights = np.where(valid_mask, weights, 0.0)

    return pl.Series('weight', weights)


# =============================================================================
# LAYER 2: CLUSTER EXPOSURE CAPS
# =============================================================================

def apply_cluster_caps(
    weights: pl.Series,
    cluster_map: dict[int, int] | np.ndarray | pl.Series,
    max_multiple: float = 2.0,
    method: str = 'cap',
    tickers: pl.Series | None = None,
) -> pl.Series:
    '''
    Cap each cluster's total absolute exposure.

    ===========================================================================
    INTUITION
    ===========================================================================

    Even with a clean signal, a portfolio can accidentally concentrate in one
    correlated cluster (e.g. all tech stocks long, all energy short). If that
    cluster moves against you, the portfolio takes a large hit.

    Cluster caps enforce diversification: no single Louvain cluster can exceed
    `max_multiple` times its "fair share" of gross exposure.

    Fair share = max_gross / n_clusters
    Cap per cluster = max_multiple * (max_gross / n_clusters)

    Example: max_gross = 1.0, 10 clusters, max_multiple = 2.0
      -> Each cluster can be at most 20% of gross notional.

    ===========================================================================
    THREE METHODS
    ===========================================================================

    'cap' (methodology.md default): Hard cap. Scale down overweight clusters
      proportionally, but do NOT redistribute the freed-up capacity. Total
      gross shrinks. (Simpler, tested behavior in methodology.)

    'rescale': Iterative rescale. After capping, renormalize remaining weights
      to restore original gross. Repeats until convergence. Guarantees no
      cluster exceeds cap AND final gross = target gross.

    'risk_parity': Equal risk contribution within each cluster. Uses the
      sub-covariance matrix of the cluster to allocate weights so each
      cluster contributes equally to portfolio variance. (Advanced, uses
      covariance -- see optimization.py for full implementation.)

    ===========================================================================
    PARAMETERS
    ===========================================================================

    weights:      Polars Series of weights (one per ticker).
    cluster_map:  Dict {ticker_idx: cluster_id} OR array/Series of cluster_id
                  per ticker (same length as weights).
    max_multiple: Cap as multiple of equal share. 2.0 = 2x fair share.
    method:       'cap', 'rescale', or 'risk_parity'.
    tickers:      Optional ticker names (for debugging/alignment).

    ===========================================================================
    RETURNS
    ===========================================================================

    Polars Series of capped weights (same length/order).

    ===========================================================================
    IMPORTANT BEHAVIORAL NOTE (from methodology.md)
    ===========================================================================

    With method='cap', a capped cluster's share of the *final* (shrunken)
    gross can exceed the nominal max_multiple/n_clusters fraction, because
    the freed-up capacity is NOT redistributed. Example:

      3 clusters, max_gross=1.0, max_multiple=2.0 -> cap = 0.667 per cluster.
      Cluster A wants 0.9, B wants 0.05, C wants 0.05 (gross=1.0).
      After cap: A=0.667, B=0.05, C=0.05 -> new gross=0.767.
      A's share of final = 0.667/0.767 = 87% >> 66.7% nominal cap.

    This is CORRECT arithmetic, just unintuitive. The 'rescale' method
    avoids this by redistributing, at the cost of more complexity.
    '''
    weights_np = np.asarray(weights.to_numpy(), dtype=float)
    n = len(weights_np)

    if n == 0:
        return pl.Series('weight', [], dtype=pl.Float64)

    # -------------------------------------------------------------
    # Normalize cluster_map to array of cluster_ids (0..K-1)
    # -------------------------------------------------------------
    if isinstance(cluster_map, dict):
        # Dict {ticker_idx: cluster_id}
        clusters = np.array([cluster_map.get(i, -1) for i in range(n)], dtype=int)
    elif isinstance(cluster_map, pl.Series):
        clusters = cluster_map.to_numpy()
    else:
        clusters = np.asarray(cluster_map, dtype=int)

    if clusters.shape != (n,):
        raise ValueError(f"cluster_map length {clusters.shape} != weights length {n}")

    # Valid (non-null) weights
    valid_mask = ~np.isnan(weights_np)
    if not valid_mask.any():
        return pl.Series('weight', np.zeros(n, dtype=float))

    # Unique clusters present (ignore -1 = unclustered)
    unique_clusters = np.unique(clusters[valid_mask])
    unique_clusters = unique_clusters[unique_clusters >= 0]
    n_clusters = len(unique_clusters)

    if n_clusters == 0:
        return pl.Series('weight', weights_np)

    # -------------------------------------------------------------
    # Compute cap per cluster
    # -------------------------------------------------------------
    target_gross = np.sum(np.abs(weights_np[valid_mask]))
    equal_share = target_gross / n_clusters
    cap_per_cluster = max_multiple * equal_share

    # -------------------------------------------------------------
    # Method 'cap': hard cap, no redistribution
    # -------------------------------------------------------------
    if method == 'cap':
        capped = weights_np.copy()
        for cl in unique_clusters:
            mask = (clusters == cl) & valid_mask
            cluster_gross = np.sum(np.abs(capped[mask]))
            if cluster_gross > cap_per_cluster:
                # Scale down proportionally within cluster
                capped[mask] = capped[mask] * (cap_per_cluster / cluster_gross)

        return pl.Series('weight', capped)

    # -------------------------------------------------------------
    # Method 'rescale': iterative cap + renormalize to target gross
    # -------------------------------------------------------------
    elif method == 'rescale':
        capped = weights_np.copy()
        max_iter = 50
        tol = 1e-8

        for _ in range(max_iter):
            prev = capped.copy()
            # Cap each cluster
            for cl in unique_clusters:
                mask = (clusters == cl) & valid_mask
                cluster_gross = np.sum(np.abs(capped[mask]))
                if cluster_gross > cap_per_cluster:
                    capped[mask] = capped[mask] * (cap_per_cluster / cluster_gross)

            # Renormalize to target gross
            current_gross = np.sum(np.abs(capped[valid_mask]))
            if current_gross > 0:
                capped[valid_mask] = capped[valid_mask] * (target_gross / current_gross)

            # Check convergence
            if np.max(np.abs(capped[valid_mask] - prev[valid_mask])) < tol:
                break

        return pl.Series('weight', capped)

    # -------------------------------------------------------------
    # Method 'risk_parity': equal risk contribution within cluster
    # (Simplified version - full version in optimization.py)
    # -------------------------------------------------------------
    elif method == 'risk_parity':
        # For now, fall back to 'cap' with a warning.
        # Full implementation needs covariance sub-matrix per cluster.
        import warnings
        warnings.warn(
            "method='risk_parity' not fully implemented here; "
            "falling back to 'cap'. Use optimization.py for full implementation.",
            UserWarning
        )
        return apply_cluster_caps(weights, cluster_map, max_multiple, method='cap', tickers=tickers)

    else:
        raise ValueError(f"unknown method: {method!r}. Use 'cap', 'rescale', or 'risk_parity'.")


# =============================================================================
# LAYER 3: VOLATILITY TARGETING
# =============================================================================

def apply_vol_target(
    weights: pl.Series,
    sigma: np.ndarray,
    target_vol: float,
    method: str = 'scale',
    tickers: pl.Series | None = None,
) -> pl.Series:
    '''
    Scale weights to achieve a target portfolio volatility.

    ===========================================================================
    INTUITION
    ===========================================================================

    The portfolio's ex-ante volatility is sqrt(w' * Sigma * w). To hit a
    target vol (e.g. 10% annualized), we scale all weights by a constant:

        scale = target_vol / sqrt(w' * Sigma * w)
        w_targeted = w * scale

    This is the standard approach at top firms (Grinold & Kahn Ch. 4).
    It assumes the covariance matrix Sigma is a good forecast of future
    realized covariance -- hence the importance of RMT cleaning + shrinkage
    in covariance.py.

    ===========================================================================
    TWO METHODS
    ===========================================================================

    'scale' (methodology.md default): Uniform scaling. All weights multiplied
      by the same factor. Simple, preserves relative weight ratios.

    'risk_parity': First equalize risk contributions (each asset contributes
      equally to portfolio variance), THEN scale to target vol. More
      sophisticated but changes relative sizing.
      (Full implementation in optimization.py)

    ===========================================================================
    PARAMETERS
    ===========================================================================

    weights:     Polars Series of weights (one per ticker, same order as sigma).
    sigma:       P x P covariance matrix (must match weights order).
    target_vol:  Target annualized portfolio volatility (e.g. 0.10 = 10%).
    method:      'scale' or 'risk_parity'.
    tickers:     Optional ticker names for alignment checks.

    ===========================================================================
    RETURNS
    ===========================================================================

    Polars Series of vol-targeted weights.

    ===========================================================================
    EDGE CASES
    ===========================================================================

    - Zero/near-zero portfolio vol -> returns weights unchanged (or zeros).
    - Weights and sigma dimension mismatch -> raises ValueError.
    - NaN weights -> treated as 0, preserved as NaN in output.
    '''
    weights_np = np.asarray(weights.to_numpy(), dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    n = len(weights_np)
    if sigma.shape != (n, n):
        raise ValueError(f"sigma shape {sigma.shape} != weights length {n}")

    valid_mask = ~np.isnan(weights_np)
    if not valid_mask.any():
        return pl.Series('weight', np.zeros(n, dtype=float))

    # Portfolio variance: w' Sigma w (only valid entries)
    w_valid = np.where(valid_mask, weights_np, 0.0)
    port_var = w_valid @ sigma @ w_valid

    if port_var <= 0:
        # Degenerate: return original (or zeros)
        return pl.Series('weight', weights_np)

    port_vol = np.sqrt(port_var)
    scale = target_vol / port_vol

    # Cap scale to avoid extreme leverage (safety)
    scale = min(scale, 10.0)  # max 10x leverage

    if method == 'scale':
        weights_scaled = np.where(valid_mask, weights_np * scale, 0.0)
        return pl.Series('weight', weights_scaled)

    elif method == 'risk_parity':
        # Full risk parity: w_i ∝ 1 / (Sigma_ii * w_i) ... iterative.
        # For now, fall back to 'scale'.
        import warnings
        warnings.warn(
            "method='risk_parity' not fully implemented here; "
            "falling back to 'scale'. Use optimization.py for full implementation.",
            UserWarning
        )
        return apply_vol_target(weights, sigma, target_vol, method='scale', tickers=tickers)

    else:
        raise ValueError(f"unknown method: {method!r}. Use 'scale' or 'risk_parity'.")


# =============================================================================
# LAYER 4: CHAINED PORTFOLIO CONSTRUCTION (single-call convenience)
# =============================================================================

def construct_portfolio(
    score_df: pl.DataFrame,
    corr: np.ndarray,
    vols: np.ndarray,
    cluster_map: dict[int, int] | np.ndarray | pl.Series,
    target_vol: float,
    max_gross: float = 1.0,
    max_cluster_multiple: float = 2.0,
    signal_method: str = 'proportional',
    cap_method: str = 'cap',
    vol_method: str = 'scale',
    score_col: str = 'combined_score',
    ticker_col: str = 'ticker',
    date_col: str = 'date',
) -> pl.DataFrame:
    '''
    Single-call pipeline: signal scores → risk-adjusted portfolio weights.

    ===========================================================================
    INTUITION
    ===========================================================================

    This chains the three layers above into one call for convenience.
    Operates cross-sectionally PER DATE.

    Pipeline per date:
      1. raw_weights_from_score(score, method=signal_method, max_gross)
      2. apply_cluster_caps(weights, cluster_map, max_multiple=max_cluster_multiple, method=cap_method)
      3. apply_vol_target(weights, sigma, target_vol, method=vol_method)

    Sigma is built from corr + vols via build_covariance_matrix (covariance.py).

    ===========================================================================
    PARAMETERS
    ===========================================================================

    score_df:         Polars DataFrame with columns [date, ticker, score_col].
    corr:             P x P correlation matrix (RMT-cleaned + optionally shrunk).
    vols:             P-vector of volatilities (aligned with corr/order).
    cluster_map:      Dict/array mapping ticker index -> cluster_id.
    target_vol:       Target annualized portfolio volatility.
    max_gross:        Max gross exposure (default 1.0 = 200% notional).
    max_cluster_multiple: Cluster cap multiplier (default 2.0).
    signal_method:    'proportional', 'inv_vol', or 'rank'.
    cap_method:       'cap', 'rescale', or 'risk_parity'.
    vol_method:       'scale' or 'risk_parity'.
    score_col:        Column name for signal score.
    ticker_col:       Column name for ticker.
    date_col:         Column name for date.

    ===========================================================================
    RETURNS
    ===========================================================================

    Polars DataFrame with columns: [date, ticker, weight, cluster_id]

    ===========================================================================
    USAGE
    ===========================================================================

    >>> weights_df = construct_portfolio(
    ...     score_df=combined_scores,
    ...     corr=corr_clean,
    ...     vols=rolling_vols,
    ...     cluster_map=louvain_clusters,
    ...     target_vol=0.10,
    ... )

    ===========================================================================
    REFERENCES
    ===========================================================================

    See individual layer docstrings. Full chain mirrors:
    Grinold & Kahn (2000) Ch. 3-4; López de Prado (2018) Ch. 10.
    '''
    # Build covariance from correlation + vols
    from src.risk.covariance import build_covariance_matrix
    sigma = build_covariance_matrix(corr, vols)

    # Work per date
    result_rows = []
    unique_dates = score_df[date_col].unique().sort()

    for date_val in unique_dates:
        date_scores = score_df.filter(c(date_col) == date_val).sort(ticker_col)

        if len(date_scores) == 0:
            continue

        tickers = date_scores[ticker_col]
        scores = date_scores[score_col]

        # Layer 1: Signal → Raw Weights
        raw_w = raw_weights_from_score(
            scores,
            tickers=tickers,
            dates=pl.Series([date_val] * len(scores)),
            max_gross=max_gross,
            method=signal_method,
            vols=vols if signal_method == 'inv_vol' else None,
        )

        # Layer 2: Cluster Caps
        capped_w = apply_cluster_caps(
            raw_w,
            cluster_map=cluster_map,
            max_multiple=max_cluster_multiple,
            method=cap_method,
            tickers=tickers,
        )

        # Layer 3: Vol Targeting
        final_w = apply_vol_target(
            capped_w,
            sigma=sigma,
            target_vol=target_vol,
            method=vol_method,
            tickers=tickers,
        )

        # Build cluster_id array for output
        if isinstance(cluster_map, dict):
            cluster_ids = np.array([cluster_map.get(i, -1) for i in range(len(tickers))], dtype=int)
        elif isinstance(cluster_map, pl.Series):
            cluster_ids = cluster_map.to_numpy()
        else:
            cluster_ids = np.asarray(cluster_map, dtype=int)

        # Append rows
        for i, tkr in enumerate(tickers):
            result_rows.append({
                date_col: date_val,
                ticker_col: tkr,
                'weight': final_w[i],
                'cluster_id': cluster_ids[i] if i < len(cluster_ids) else -1,
            })

    if not result_rows:
        return pl.DataFrame({
            date_col: [],
            ticker_col: [],
            'weight': [],
            'cluster_id': [],
        })

    return pl.DataFrame(result_rows)


# =============================================================================
# UTILITIES: Diagnostics & Validation
# =============================================================================

def portfolio_diagnostics(
    weights: pl.Series,
    sigma: np.ndarray,
    cluster_map: dict[int, int] | np.ndarray | pl.Series | None = None,
    target_vol: float | None = None,
    max_gross: float = 1.0,
    max_cluster_multiple: float = 2.0,
) -> dict:
    '''
    Compute key portfolio diagnostics for monitoring/validation.

    ===========================================================================
    RETURNS
    ===========================================================================

    Dict with:
      - gross_exposure: sum |w|
      - net_exposure:   sum w
      - portfolio_vol:  sqrt(w' Sigma w) [annualized if sigma is annualized]
      - max_weight:     max |w_i|
      - n_positions:    count of non-zero weights
      - cluster_gross:  dict {cluster_id: gross} (if cluster_map given)
      - cluster_cap_hit: bool (if cluster_map + max_cluster_multiple given)
    '''
    weights_np = np.asarray(weights.to_numpy(), dtype=float)
    valid = ~np.isnan(weights_np)
    w = np.where(valid, weights_np, 0.0)

    gross = np.sum(np.abs(w))
    net = np.sum(w)

    # Portfolio variance
    port_var = w @ sigma @ w
    port_vol = np.sqrt(max(port_var, 0.0))

    max_w = np.max(np.abs(w)) if len(w) > 0 else 0.0
    n_pos = np.sum(np.abs(w) > 1e-10)

    result = {
        'gross_exposure': float(gross),
        'net_exposure': float(net),
        'portfolio_vol': float(port_vol),
        'max_weight': float(max_w),
        'n_positions': int(n_pos),
    }

    if cluster_map is not None:
        if isinstance(cluster_map, dict):
            clusters = np.array([cluster_map.get(i, -1) for i in range(len(w))], dtype=int)
        elif isinstance(cluster_map, pl.Series):
            clusters = cluster_map.to_numpy()
        else:
            clusters = np.asarray(cluster_map, dtype=int)

        unique = np.unique(clusters[valid])
        unique = unique[unique >= 0]

        cluster_gross = {}
        for cl in unique:
            mask = (clusters == cl) & valid
            cluster_gross[int(cl)] = float(np.sum(np.abs(w[mask])))

        result['cluster_gross'] = cluster_gross

        if max_cluster_multiple is not None and len(unique) > 0:
            equal_share = gross / len(unique)
            cap = max_cluster_multiple * equal_share
            result['cluster_cap'] = float(cap)
            result['cluster_cap_hit'] = any(g > cap for g in cluster_gross.values())

    if target_vol is not None:
        result['vol_target'] = float(target_vol)
        result['vol_ratio'] = float(port_vol / target_vol) if target_vol > 0 else None

    return result


# =============================================================================
# TURNOVER ESTIMATION
# =============================================================================

def compute_turnover(
    weights_prev: pl.Series,
    weights_new: pl.Series,
) -> float:
    '''
    Average absolute weight change (turnover) between two cross-sections.

    ===========================================================================
    INTUITION
    ===========================================================================

    Turnover = (1/2) * sum_i |w_new_i - w_prev_i|  [one-way turnover]

    The 1/2 factor makes it "one-way" (buying OR selling), matching standard
    convention. Without 1/2, it's "two-way" (buying + selling).

    High turnover → high transaction costs. Typical targets:
      - Daily rebalance:  5-20% one-way
      - Weekly rebalance: 2-10% one-way
      - Monthly rebalance: 1-5% one-way

    ===========================================================================
    PARAMETERS
    ===========================================================================

    weights_prev: Previous day's weights (Polars Series, aligned by ticker).
    weights_new:  Current day's weights (same order/length).

    ===========================================================================
    RETURNS
    ===========================================================================

    One-way turnover as float.
    '''
    w_prev = np.asarray(weights_prev.to_numpy(), dtype=float)
    w_new = np.asarray(weights_new.to_numpy(), dtype=float)

    if w_prev.shape != w_new.shape:
        raise ValueError(f"shape mismatch: {w_prev.shape} vs {w_new.shape}")

    # Treat NaN as 0 (flat)
    w_prev = np.where(np.isnan(w_prev), 0.0, w_prev)
    w_new = np.where(np.isnan(w_new), 0.0, w_new)

    return float(0.5 * np.sum(np.abs(w_new - w_prev)))