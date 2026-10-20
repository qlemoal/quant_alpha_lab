'''
src/portfolio/costs.py

Transaction cost models for realistic portfolio simulation.

================================================================================
WHY THIS MATTERS
================================================================================

A backtest without transaction costs is a fantasy. Every top quant fund models
costs explicitly because they are the difference between a theoretical Sharpe
and a realizable one. Cost models are also used *inside* the optimizer
(cost-aware optimization) to avoid trading when the marginal alpha doesn't
justify the marginal cost.

The standard decomposition (used at AQR, Citadel, Two Sigma, etc.):

  Total Cost = Spread Cost + Commission + Market Impact

  1. SPREAD:   Half the bid-ask spread paid on each trade.
                 Cost = 0.5 * spread * |Δw| * notional
                 Spread ≈ f(ADV, volatility) -- wider for illiquid/small names.

  2. COMMISSION: Fixed per-share or per-notional fee.
                 Cost = commission_bps * |Δw| * notional

  3. MARKET IMPACT: Price moves against you because you trade.
                 Cost = impact_coef * σ * √(|Δw| / ADV) * |Δw| * notional
                 (Square-root model: Almgren-Chriss 2001)

The square-root impact model is the industry standard:
  - Theory: Kyle (1985), Almgren-Chriss (2001)
  - Empirical: Engle, Ferstenberg, Russell (2012), Bacry et al. (2015)
  - Impact ∝ √(participation_rate) where participation = |Δw| * notional / ADV

================================================================================
INTEGRATION
================================================================================

Consumed by:
  - src/portfolio/backtest.py       -- applies costs to realized returns
  - src/portfolio/optimization.py   -- cost-aware optimization objective

Inputs from existing pipeline:
  - ADV:          src/features/adv (average dollar volume)
  - Volatility:   src/features/volatility (rolling std)
  - Weights:      src/risk/portfolio_construction (daily weights)

================================================================================
REFERENCES
================================================================================

  - Almgren, R., & Chriss, N. (2001). "Optimal execution of portfolio transactions."
    Journal of Risk, 3(2), 5-39.  (Square-root impact model.)

  - Kyle, A. S. (1985). "Continuous auctions and insider trading."
    Econometrica, 53(6), 1315-1335.  (Market impact theory.)

  - Engle, R., Ferstenberg, R., & Russell, J. (2012). "Measuring and modeling
    execution cost and market impact." Journal of Financial Markets, 15(2), 147-172.

  - Bacry, E., Iuga, A., Lasnier, M., & Bouchaud, J. P. (2015). "Market impact
    and trading profile of hidden orders in order book markets."

  - Frazzini, A., Israel, R., & Moskowitz, T. J. (2018). "Trading costs."
    AQR Capital Management working paper.  (Empirical cost estimates.)

  - Garleanu, N., & Pedersen, L. H. (2013). "Dynamic trading with predictable
    returns and transaction costs." Journal of Finance, 68(6), 2309-2340.
    (Cost-aware dynamic optimization.)

================================================================================
'''

import numpy as np
import polars as pl
from polars import col as c


# =============================================================================
# SPREAD ESTIMATION
# =============================================================================

def estimate_spread(
    adv: np.ndarray,
    vol: np.ndarray | None = None,
    model: str = 'power_law',
    spread_bps_min: float = 1.0,
    spread_bps_max: float = 200.0,
    **kwargs,
) -> np.ndarray:
    '''
    Estimate half-spread (cost to cross the spread) in basis points.

    ===========================================================================
    INTUITION
    ===========================================================================

    The bid-ask spread is the most direct transaction cost. It varies
    systematically:

      - High ADV (liquid) → tight spread
      - Low ADV (illiquid) → wide spread
      - High volatility → wider spread (market makers need more compensation)

    Two standard models:

    1. POWER LAW (Bacry et al. 2015):
       spread_bps = a * ADV^(-b) * (vol^c)
       Typical: a ~ 100-200, b ~ 0.2-0.3, c ~ 0.5

    2. LIQUIDITY BUCKETS:
       Bin ADV into quintiles/deciles, assign fixed spread per bucket.
       Simpler, more robust, used in production at many firms.

    ===========================================================================
    PARAMETERS
    ===========================================================================

    adv:           P-vector of average daily dollar volume (same units as notional).
    vol:           Optional P-vector of volatilities (for vol-adjusted model).
    model:         'power_law', 'bucket', or 'fixed'.
    spread_bps_min: Floor (bps) -- no spread tighter than this.
    spread_bps_max: Cap (bps) -- no spread wider than this.
    **kwargs:      Model-specific parameters (see below).

    POWER_LAW kwargs:
      a:       scale (default 150)
      b:       ADV exponent (default 0.25)
      c:       vol exponent (default 0.5, requires vol)

    BUCKET kwargs:
      n_buckets:    number of ADV quantile buckets (default 10)
      spreads_bps:  list of spread_bps per bucket (length n_buckets)

    FIXED kwargs:
      spread_bps:   constant spread for all assets (default 10)

    ===========================================================================
    RETURNS
    ===========================================================================

    P-vector of half-spreads in basis points (0.01% = 1 bps).
    '''
    adv = np.asarray(adv, dtype=float)
    n = len(adv)

    if model == 'power_law':
        a = kwargs.get('a', 150.0)
        b = kwargs.get('b', 0.25)
        c = kwargs.get('c', 0.5)

        # Avoid division by zero / log of zero
        adv_safe = np.maximum(adv, 1.0)
        spread = a * (adv_safe ** (-b))

        if vol is not None:
            vol = np.asarray(vol, dtype=float)
            vol_safe = np.maximum(vol, 1e-6)
            spread = spread * (vol_safe ** c)

    elif model == 'bucket':
        n_buckets = kwargs.get('n_buckets', 10)
        spreads_bps = kwargs.get('spreads_bps', None)

        if spreads_bps is None:
            # Default: geometric progression from 5bps (most liquid) to 50bps
            spreads_bps = np.geomspace(5.0, 50.0, n_buckets)

        spreads_bps = np.asarray(spreads_bps)
        if len(spreads_bps) != n_buckets:
            raise ValueError(f"spreads_bps length {len(spreads_bps)} != n_buckets {n_buckets}")

        # Assign to buckets by ADV quantile
        # Use percentiles to handle outliers
        percentiles = np.percentile(adv, np.linspace(0, 100, n_buckets + 1))
        # Handle edge case where all ADV equal
        if np.allclose(percentiles[0], percentiles[-1]):
            spread = np.full(n, spreads_bps[0])
        else:
            bucket_idx = np.digitize(adv, percentiles[1:-1])
            bucket_idx = np.clip(bucket_idx, 0, n_buckets - 1)
            spread = spreads_bps[bucket_idx]

    elif model == 'fixed':
        spread = np.full(n, kwargs.get('spread_bps', 10.0))

    else:
        raise ValueError(f"unknown model: {model!r}. Use 'power_law', 'bucket', or 'fixed'.")

    # Clip to bounds
    spread = np.clip(spread, spread_bps_min, spread_bps_max)

    return spread


# =============================================================================
# MARKET IMPACT ESTIMATION
# =============================================================================

def estimate_market_impact(
    weight_change: np.ndarray,
    adv: np.ndarray,
    vol: np.ndarray,
    notional: float,
    model: str = 'sqrt',
    **kwargs,
) -> np.ndarray:
    '''
    Estimate market impact cost per asset in dollars.

    ===========================================================================
    INTUITION
    ===========================================================================

    Market impact is the adverse price movement caused by your own trading.
    The square-root model (Almgren-Chriss 2001) is the industry standard:

      Impact (in $) = κ * σ * √(participation) * |Δw| * notional

    where:
      - κ: impact coefficient (typically 0.1 - 1.0, calibrated)
      - σ: daily volatility (as decimal, e.g. 0.02 = 2%)
      - participation = |Δw| * notional / ADV
      - Δw: weight change (target - current)
      - notional: total portfolio value ($)

    Intuition: if you trade 1% of ADV, your participation is 1%. Impact grows
    with the SQUARE ROOT of participation, not linearly -- this reflects the
    concave liquidity supply curve in limit order books.

    ===========================================================================
    PARAMETERS
    ===========================================================================

    weight_change:  P-vector of weight changes (target - current).
    adv:            P-vector of ADV ($).
    vol:            P-vector of daily volatilities (decimal).
    notional:       Portfolio notional value ($).
    model:          'sqrt' (Almgren-Chriss), 'linear' (Kyle), or 'almgren_chriss_full'.
    **kwargs:
      - kappa:         impact coefficient (default 1.0 for sqrt, calibrated)
      - tau:           temporary impact fraction (default 0.5, for full AC)
      - sigma_daily:   if vol is annualized, pass sigma_daily=True

    ===========================================================================
    RETURNS
    ===========================================================================

    P-vector of impact costs in DOLLARS (same sign as weight_change,
    always positive cost).
    '''
    weight_change = np.asarray(weight_change, dtype=float)
    adv = np.asarray(adv, dtype=float)
    vol = np.asarray(vol, dtype=float)
    notional = float(notional)

    n = len(weight_change)
    if adv.shape != (n,) or vol.shape != (n,):
        raise ValueError("adv, vol must have same length as weight_change")

    abs_dw = np.abs(weight_change)

    # Participation rate = |Δw| * notional / ADV
    participation = np.zeros(n)
    mask = adv > 0
    participation[mask] = abs_dw[mask] * notional / adv[mask]
    participation = np.clip(participation, 0.0, 1.0)  # cap at 100%

    if model == 'sqrt':
        # Standard square-root model
        kappa = kwargs.get('kappa', 1.0)
        impact = kappa * vol * np.sqrt(participation) * abs_dw * notional

    elif model == 'linear':
        # Kyle (1985) linear impact: impact ∝ participation
        kappa = kwargs.get('kappa', 1.0)
        impact = kappa * vol * participation * abs_dw * notional

    elif model == 'almgren_chriss_full':
        # Full Almgren-Chriss: temporary + permanent impact
        # Temporary: κ * σ * √(participation) * |Δw|
        # Permanent: 0.5 * κ * σ * participation * |Δw|
        kappa = kwargs.get('kappa', 1.0)
        tau = kwargs.get('tau', 0.5)  # temporary fraction
        temp_impact = kappa * vol * np.sqrt(participation) * abs_dw
        perm_impact = 0.5 * kappa * vol * participation * abs_dw
        impact = (tau * temp_impact + (1 - tau) * perm_impact) * notional

    else:
        raise ValueError(f"unknown model: {model!r}. Use 'sqrt', 'linear', or 'almgren_chriss_full'.")

    return impact


# =============================================================================
# COMMISSION
# =============================================================================

def estimate_commission(
    weight_change: np.ndarray,
    notional: float,
    commission_bps: float = 1.0,
    min_commission_per_share: float = 0.0,
    prices: np.ndarray | None = None,
) -> np.ndarray:
    '''
    Estimate commission cost per asset in dollars.

    ===========================================================================
    INTUITION
    ===========================================================================

    Commission is the broker fee. Two common structures:

    1. PER-NOTIONAL (bps): Cost = commission_bps * |Δw| * notional
       Simple, standard for institutional brokers.

    2. PER-SHARE: Cost = min_commission_per_share * |Δshares|
       Where Δshares = |Δw| * notional / price

    ===========================================================================
    PARAMETERS
    ===========================================================================

    weight_change:  P-vector of weight changes.
    notional:       Portfolio notional ($).
    commission_bps: Commission in basis points of notional traded.
    min_commission_per_share: Per-share commission (if using per-share model).
    prices:         Required if using per-share model. P-vector of prices.

    ===========================================================================
    RETURNS
    ===========================================================================

    P-vector of commission costs in DOLLARS.
    '''
    weight_change = np.asarray(weight_change, dtype=float)
    abs_dw = np.abs(weight_change)

    if min_commission_per_share > 0:
        if prices is None:
            raise ValueError("prices required for per-share commission model")
        prices = np.asarray(prices, dtype=float)
        shares_traded = abs_dw * notional / np.maximum(prices, 1e-6)
        commission = min_commission_per_share * shares_traded
    else:
        commission = (commission_bps / 10000.0) * abs_dw * notional

    return commission


# =============================================================================
# TOTAL TRANSACTION COST
# =============================================================================

def total_transaction_cost(
    weights_prev: np.ndarray,
    weights_new: np.ndarray,
    adv: np.ndarray,
    vol: np.ndarray,
    notional: float,
    prices: np.ndarray | None = None,
    spread_model: str = 'power_law',
    spread_kwargs: dict | None = None,
    impact_model: str = 'sqrt',
    impact_kwargs: dict | None = None,
    commission_bps: float = 1.0,
    min_commission_per_share: float = 0.0,
) -> tuple[float, dict]:
    '''
    Complete transaction cost for a portfolio rebalance.

    ===========================================================================
    INTUITION
    ===========================================================================

    One-stop function: computes spread + impact + commission for the full
    portfolio transition from weights_prev to weights_new.

    Returns both the total cost (scalar) and a detailed breakdown per asset.

    ===========================================================================
    PARAMETERS
    ===========================================================================

    weights_prev:  P-vector of current weights.
    weights_new:   P-vector of target weights.
    adv:           P-vector of ADV ($).
    vol:           P-vector of daily vols (decimal).
    notional:      Portfolio notional ($).
    prices:        P-vector of prices ($) -- needed for per-share commission.
    spread_model, spread_kwargs:   Passed to estimate_spread.
    impact_model, impact_kwargs:   Passed to estimate_market_impact.
    commission_bps:                Per-notional commission (bps).
    min_commission_per_share:      Per-share commission ($).

    ===========================================================================
    RETURNS
    ===========================================================================

    (total_cost_dollars, breakdown_dict):
        breakdown_dict = {
            'spread':     P-vector,
            'impact':     P-vector,
            'commission': P-vector,
            'total':      P-vector (sum of above),
            'by_asset':   list of dicts with all components per asset,
        }
    '''
    weights_prev = np.asarray(weights_prev, dtype=float)
    weights_new = np.asarray(weights_new, dtype=float)
    dw = weights_new - weights_prev
    abs_dw = np.abs(dw)

    n = len(dw)

    # 1. Spread cost (half-spread * notional traded)
    spread_bps = estimate_spread(adv, vol, model=spread_model, **(spread_kwargs or {}))
    spread_cost = (spread_bps / 10000.0) * abs_dw * notional  # half-spread

    # 2. Market impact
    impact_cost = estimate_market_impact(dw, adv, vol, notional, model=impact_model, **(impact_kwargs or {}))

    # 3. Commission
    commission_cost = estimate_commission(dw, notional, commission_bps, min_commission_per_share, prices)

    # Total
    total_per_asset = spread_cost + impact_cost + commission_cost
    total_cost = float(np.sum(total_per_asset))

    breakdown = {
        'spread': spread_cost,
        'impact': impact_cost,
        'commission': commission_cost,
        'total': total_per_asset,
        'by_asset': [
            {
                'asset_idx': i,
                'weight_change': float(dw[i]),
                'spread': float(spread_cost[i]),
                'impact': float(impact_cost[i]),
                'commission': float(commission_cost[i]),
                'total': float(total_per_asset[i]),
            }
            for i in range(n)
        ],
    }

    return total_cost, breakdown


# =============================================================================
# COST-ADJUSTED RETURNS (for backtesting)
# =============================================================================

def cost_adjusted_returns(
    gross_returns: np.ndarray,
    weights: np.ndarray,
    weights_prev: np.ndarray,
    adv: np.ndarray,
    vol: np.ndarray,
    notional: float,
    prices: np.ndarray | None = None,
    spread_model: str = 'power_law',
    spread_kwargs: dict | None = None,
    impact_model: str = 'sqrt',
    impact_kwargs: dict | None = None,
    commission_bps: float = 1.0,
    min_commission_per_share: float = 0.0,
) -> tuple[np.ndarray, float]:
    '''
    Apply transaction costs to gross returns to get net returns.

    ===========================================================================
    INTUITION
    ===========================================================================

    In a backtest, you have:
      - Gross daily returns of the portfolio (from weights * asset returns)
      - Daily weight changes (from rebalancing)

    Net return = Gross return - (Transaction cost / Notional)

    This function computes the cost of transitioning from weights_prev to
    weights, and subtracts it from the gross return for that day.

    ===========================================================================
    PARAMETERS
    ===========================================================================

    gross_returns:  Daily portfolio gross returns (scalar or T-vector).
    weights:        T x P matrix of daily weights.
    weights_prev:   T x P matrix of previous day's weights (shifted).
    adv, vol:       P-vectors (or T x P if time-varying).
    notional:       Portfolio notional ($).
    ...             Cost model parameters.

    ===========================================================================
    RETURNS
    ===========================================================================

    (net_returns, total_costs):
        net_returns:  T-vector of net returns.
        total_costs:  T-vector of daily transaction costs ($).
    '''
    gross_returns = np.asarray(gross_returns, dtype=float)
    weights = np.asarray(weights, dtype=float)
    weights_prev = np.asarray(weights_prev, dtype=float)

    if gross_returns.ndim == 0:
        gross_returns = np.array([gross_returns])
    T = len(gross_returns)

    if weights.shape[0] != T or weights_prev.shape[0] != T:
        raise ValueError("weights/weights_prev must have T rows matching gross_returns")

    n_assets = weights.shape[1]

    # Handle ADV/vol: can be P-vector (static) or T x P (time-varying)
    if adv.ndim == 1:
        adv_t = np.tile(adv, (T, 1))
    else:
        adv_t = adv

    if vol.ndim == 1:
        vol_t = np.tile(vol, (T, 1))
    else:
        vol_t = vol

    net_returns = np.zeros(T)
    total_costs = np.zeros(T)

    for t in range(T):
        dw = weights[t] - weights_prev[t]
        _, breakdown = total_transaction_cost(
            weights_prev[t], weights[t],
            adv_t[t], vol_t[t], notional,
            prices=prices[t] if prices is not None and prices.ndim == 2 else prices,
            spread_model=spread_model,
            spread_kwargs=spread_kwargs,
            impact_model=impact_model,
            impact_kwargs=impact_kwargs,
            commission_bps=commission_bps,
            min_commission_per_share=min_commission_per_share,
        )
        cost = breakdown['total'].sum()
        total_costs[t] = cost
        net_returns[t] = gross_returns[t] - cost / notional

    return net_returns, total_costs


# =============================================================================
# TURNOVER & COST ANALYSIS
# =============================================================================

def turnover_cost_analysis(
    weights_series: list[np.ndarray],
    adv: np.ndarray,
    vol: np.ndarray,
    notional: float,
    prices: np.ndarray | None = None,
    **cost_kwargs,
) -> dict:
    '''
    Analyze turnover and costs across a weight trajectory.

    ===========================================================================
    RETURNS
    ===========================================================================

    Dict with:
      - daily_turnover:    T-vector of one-way turnover
      - daily_costs_bps:   T-vector of costs in bps of notional
      - avg_turnover:      mean daily turnover
      - avg_cost_bps:      mean daily cost (bps)
      - cost_drag_annual:  annualized cost drag (bps/year)
    '''
    T = len(weights_series)
    if T < 2:
        return {'daily_turnover': [], 'daily_costs_bps': [], 'avg_turnover': 0, 'avg_cost_bps': 0, 'cost_drag_annual': 0}

    daily_turnover = np.zeros(T - 1)
    daily_costs_bps = np.zeros(T - 1)

    for t in range(T - 1):
        w_prev = weights_series[t]
        w_new = weights_series[t + 1]
        dw = w_new - w_prev

        # Turnover
        daily_turnover[t] = 0.5 * np.sum(np.abs(dw))

        # Cost
        _, breakdown = total_transaction_cost(w_prev, w_new, adv, vol, notional, prices=prices[t] if prices is not None else None, **cost_kwargs)
        daily_costs_bps[t] = breakdown['total'].sum() / notional * 10000.0  # bps

    avg_turnover = np.mean(daily_turnover)
    avg_cost_bps = np.mean(daily_costs_bps)
    cost_drag_annual = avg_cost_bps * 252  # annualize

    return {
        'daily_turnover': daily_turnover,
        'daily_costs_bps': daily_costs_bps,
        'avg_turnover': float(avg_turnover),
        'avg_cost_bps': float(avg_cost_bps),
        'cost_drag_annual': float(cost_drag_annual),
    }


# =============================================================================
# COST-AWARE OPTIMIZATION HELPER
# =============================================================================

def cost_penalty_quadratic_form(
    weights_prev: np.ndarray,
    adv: np.ndarray,
    vol: np.ndarray,
    notional: float,
    kappa: float = 1.0,
) -> np.ndarray:
    '''
    Quadratic approximation of transaction costs for optimization.

    ===========================================================================
    INTUITION
    ===========================================================================

    For cost-aware optimization, we want a QUADRATIC cost function:
        Cost ≈ 0.5 * Δw' * Λ * Δw

    where Λ is a diagonal matrix with Λ_ii = κ * σ_i * notional / √(ADV_i)

    This comes from approximating the square-root impact:
      impact = κ * σ * √(|Δw| * notional / ADV) * |Δw| * notional

    Near the current weights, this is approximately quadratic:
      impact ≈ 0.5 * κ * σ * notional^1.5 / ADV^0.5 * (Δw)^2 / |Δw|^0.5 ...

    But the standard approach (Gârleanu & Pedersen 2013) uses:
      Cost = 0.5 * γ * Σ_i (Δw_i)^2 * σ_i * √(notional / ADV_i)

    Which is a diagonal quadratic form. We return the diagonal Λ for use
    in a QP solver.

    ===========================================================================
    RETURNS
    ===========================================================================

    P-vector of diagonal entries of the cost quadratic form matrix.
    Use as: cost = 0.5 * sum_i Λ_i * (Δw_i)^2
    '''
    adv = np.asarray(adv, dtype=float)
    vol = np.asarray(vol, dtype=float)
    adv_safe = np.maximum(adv, 1.0)
    vol_safe = np.maximum(vol, 1e-6)

    # Λ_i = κ * σ_i * notional^1.5 / √(ADV_i)
    Lambda = kappa * vol_safe * (notional ** 1.5) / np.sqrt(adv_safe)

    return Lambda


def cost_penalty_for_optimizer(
    weights_prev: np.ndarray,
    adv: np.ndarray,
    vol: np.ndarray,
    notional: float,
    kappa: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    '''
    Returns (Lambda, linear_term) for cost-aware QP:

    Minimize:  0.5 * w' Σ w - μ' w + 0.5 * (w - w_prev)' Λ (w - w_prev)

    The cost penalty adds Λ to the risk matrix and -Λ w_prev to the linear term.
    '''
    Lambda = cost_penalty_quadratic_form(weights_prev, adv, vol, notional, kappa)
    # Linear term from expanding 0.5 * (w - w_prev)' Λ (w - w_prev)
    # = 0.5 * w' Λ w - w' Λ w_prev + 0.5 * w_prev' Λ w_prev
    # The constant drops out, so we add -Λ w_prev to the linear objective.
    linear_cost = -Lambda * weights_prev

    return Lambda, linear_cost