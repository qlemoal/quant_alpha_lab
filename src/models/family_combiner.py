'''
GOAL: combine the admitted signals in a LOW-DIMENSIONAL, SIGN-FIXED way: first inside each family,
    then across families with a non-negative, shrunk Elastic Net.

WHY NOT FIT ONE COEFFICIENT PER SIGNAL?
    Admitted signals are many and highly correlated (mom60, mom120, mom252 are one bet seen three times).
    One coefficient per signal means 10 to 15 free numbers fitted on noisy returns. The weights jump
    between samples. The state of the art is the opposite: few parameters, structure fixed in advance.

THE STRUCTURE

    admitted signals  (stage 05, each with a family and a sign from src/signals/priors.py)
        |
        |  [1] ALIGN: multiply each signal by its prior sign        -> a high value always means "buy"
        |  [2] AVERAGE inside the family, equal weights             -> ONE score per family, no fitting
        v
    family scores   fam_reversal, fam_momentum, fam_low_volatility, fam_illiquidity ...   (about 2 to 5 columns)
        |
        |  [3] Elastic Net, coefficients >= 0, one-SE rule          -> a few non-negative weights
        v
    combined_score = sum_f  w_f * fam_f          (w_f = 0 means "this family adds nothing out of sample")

WHY EACH STEP
    [1] The sign comes from the economic story, not from the sample. A coefficient cannot flip with the data.
    [2] Equal weights inside a family have no estimation error. The family score is less noisy than any member.
    [3] With <= 5 inputs, the fit has little room to overfit. Non-negative weights keep the economic sign.
        The one-SE rule shrinks them further (see elastic_net_combiner.apply_one_se_rule).

THE DROP-ONE TEST (drop_one_oof_ic)
    For each family, refit WITHOUT it and compare the out-of-fold IC. If removing a family does not lower
    the IC, the family adds nothing. This is the real answer to "does this signal help?": not its own IC,
    but its marginal contribution to the combination.

No look-ahead: aligning and averaging use only the same-date values of the signals.
'''

import numpy as np
import polars as pl
from sklearn.linear_model import ElasticNet

from src.evaluation.signals.ic.core import compute_ic
from src.models.elastic_net_combiner import _prepare_X, _make_target

FAMILY_PREFIX = 'fam_'


def family_column(family: str) -> str:
    return f'{FAMILY_PREFIX}{family}'


def group_by_family(meta: dict) -> dict:
    '''
    meta: {signal: {"family": str, "sign": +1/-1}} for the ADMITTED signals.
    Returns {family: [(signal, sign), ...]} with families and members in sorted order (stable output).
    A signal without a family or a valid sign is an error: 05 must not admit it.
    '''
    out = {}
    for sig in sorted(meta):
        m = meta[sig]
        if not m or m.get('family') is None or m.get('sign') not in (1, -1):
            raise ValueError(f'{sig}: family mode needs a family and a sign (+1 or -1) from the priors. Got {m}.')
        out.setdefault(m['family'], []).append((sig, int(m['sign'])))
    return dict(sorted(out.items()))


def add_family_scores(lf: pl.LazyFrame, families: dict) -> pl.LazyFrame:
    '''
    Add one column per family: the mean of (sign * signal) over its members.
    Nulls are ignored inside the mean (a family score exists if at least one member exists on that row).
    Signals are already normalised across stocks on each date (stage 04), so they have a similar scale.
    '''
    exprs = []
    for fam, members in families.items():
        terms = [pl.col(s) * float(sign) for s, sign in members]
        exprs.append(pl.mean_horizontal(terms).alias(family_column(fam)))
    return lf.with_columns(exprs)


def _oof_ic(panel: pl.DataFrame, fit_result: dict, keep: list) -> float:
    '''
    Mean rank IC of the out-of-fold score built with only the columns at positions `keep`,
    refitted fold by fold at the SELECTED alpha, l1_ratio and sign constraint.
    '''
    model = fit_result['model']
    X = _prepare_X(panel, fit_result)[:, keep]
    y = _make_target(panel, fit_result['fwdret_col'], fit_result['target'])
    n = panel.height
    pred_sum, pred_cnt = np.zeros(n), np.zeros(n)
    for train_idx, test_idx in fit_result['cv_folds']:
        m = ElasticNet(alpha=model.alpha_, l1_ratio=model.l1_ratio_, max_iter=5000,
                       positive=fit_result.get('positive', False))
        m.fit(X[train_idx], y[train_idx])
        pred_sum[test_idx] += m.predict(X[test_idx])
        pred_cnt[test_idx] += 1
    mask = pred_cnt > 0
    scored = pl.DataFrame({
        'date': panel['date'].to_numpy()[mask],
        'ticker': panel['ticker'].to_numpy()[mask],
        'score': pred_sum[mask] / pred_cnt[mask],
        'fwd': panel[fit_result['fwdret_col']].to_numpy()[mask],
    })
    ic = compute_ic(scored.lazy(), 'score', 'fwd').collect()['ic'].drop_nans().drop_nulls()
    return float(ic.mean()) if len(ic) else float('nan')


def drop_one_oof_ic(panel: pl.DataFrame, fit_result: dict) -> dict:
    '''
    For each column (family) of the fit: out-of-fold IC with it, without it, and the drop.
        ic_full      OOF IC of the model with all columns
        ic_without   OOF IC when this column is removed (same alpha, l1_ratio, folds)
        delta        ic_full - ic_without    > 0: the column helps. <= 0: it does not.
    A model with one column cannot drop it: ic_without is nan.
    Cost: one set of fold fits per column. With <= 5 columns this is fast.
    '''
    cols = fit_result['signal_cols']
    all_idx = list(range(len(cols)))
    full = _oof_ic(panel, fit_result, all_idx)
    out = {}
    for j, name in enumerate(cols):
        rest = [k for k in all_idx if k != j]
        without = _oof_ic(panel, fit_result, rest) if rest else float('nan')
        out[name] = {'ic_full': full, 'ic_without': without,
                     'delta': full - without if np.isfinite(without) else float('nan')}
    return out