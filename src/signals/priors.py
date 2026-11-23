'''
GOAL: write down, BEFORE looking at any data, what each signal should do and why.
    This is the "economic rationale" and the "sign fixed ex ante" of the selection in stage 05.

WHY A FILE FOR THIS
    A signal that works in the data but has no story is probably a data-mining accident
    (Harvey, Liu and Zhu 2016). A signal whose sign is learned from the data can flip with the sample.
    So each signal family gets, in advance:
        family     the group it belongs to (signals of one family are the same bet, see stage 06)
        sign       +1 if a HIGH value should predict a HIGH return, -1 if a high value should predict a LOW return
        rationale  one sentence: why should this work
        reference  the paper that documents it

    Stage 05 rejects a signal with no prior (option --no_require_prior turns this off for exploration),
    and rejects a signal whose measured sign is the OPPOSITE of the prior.
    Stage 06 flips the sign with the prior, so every aligned signal has an expected positive IC.

HOW A PRIOR IS FOUND FOR A COLUMN NAME
    Column names look like  mom20_zscore_tanh, vol60_zscore_clip, log_adv20_zscore_tanh, cheat_h20_ic030.
    The transform suffix is removed, then the FIRST rule whose pattern matches is used.
    A column that matches no rule has NO prior (prior_for returns None).

    mom5_zscore_tanh        -> strip transform -> "mom5"      -> rule "mom" with window <= 21 -> reversal
    log_adv20_zscore_tanh   -> strip transform -> "log_adv20" -> rule "log_adv"               -> liquidity

THE PRIORS ARE A CHOICE, NOT A RESULT
    Write them once, commit them, and do not edit them after you saw the IC. If you change a sign after
    looking at the data, you made a data-driven choice and the hurdle in 05 is no longer honest.
    Every change must be visible in git.

References:
    Jegadeesh, N. (1990). Evidence of predictable behavior of security returns. JF, 45(3).    short-term reversal
    Jegadeesh, N. and Titman, S. (1993). Returns to buying winners and selling losers. JF, 48(1).    momentum
    Ang, A., Hodrick, R., Xing, Y. and Zhang, X. (2006). The cross-section of volatility and expected returns. JF, 61(1).
    Frazzini, A. and Pedersen, L. (2014). Betting against beta. JFE, 111(1).
    Amihud, Y. (2002). Illiquidity and stock returns. JFM, 5(1).
    Harvey, C., Liu, Y. and Zhu, H. (2016). ... and the cross-section of expected returns. RFS, 29(1).
'''

import re
from dataclasses import dataclass

# The transforms 04 can add after the feature name (see src/signals/combine.py).
TRANSFORMS = ('zscore_tanh', 'zscore_clip', 'zscore', 'rank', 'decile')

# A window of this many days or fewer is "short term". Reversal lives below about one month.
SHORT_TERM_MAX_DAYS = 21


@dataclass(frozen=True)
class Prior:
    '''What we believe about a signal before seeing data.'''
    family: str
    sign: int            # +1 or -1
    rationale: str
    reference: str


# One Prior per (kind, horizon class). Kept in a dict so the table is easy to read and to test.
PRIORS = {
    'reversal': Prior('reversal', -1,
                      'Past short-term winners are over-bought by liquidity demand and give back the move; losers rebound.',
                      'Jegadeesh (1990)'),
    'momentum': Prior('momentum', +1,
                      'Investors under-react to news, so 3 to 12 month winners keep winning for some months.',
                      'Jegadeesh and Titman (1993)'),
    'low_volatility': Prior('low_volatility', -1,
                            'Leverage-constrained investors overpay for high-volatility stocks, so these earn less.',
                            'Ang, Hodrick, Xing, Zhang (2006)'),
    'low_beta': Prior('low_beta', -1,
                      'Leverage-constrained investors overpay for high-beta stocks, so low beta earns more per unit of risk.',
                      'Frazzini and Pedersen (2014)'),
    'illiquidity': Prior('illiquidity', -1,
                         'Investors ask a premium for holding stocks that are hard to trade, so low dollar volume earns more.',
                         'Amihud (2002)'),
    # Benchmarks. They exist only so that the same gates test them. They are NOT economic priors.
    'cheat': Prior('cheat', +1,
                   'Benchmark: built from the future return with a known IC > 0. Never a strategy.',
                   'src/signals/cheat.py'),
    'placebo': Prior('placebo', +1,
                     'Benchmark: pure noise (true IC = 0). A sound pipeline must reject it.',
                     'src/signals/cheat.py'),
}


def strip_transform(name: str) -> str:
    '''mom20_zscore_tanh -> mom20 ;  log_adv20_rank -> log_adv20 ;  mom20 -> mom20.'''
    for t in TRANSFORMS:
        if name.endswith('_' + t):
            return name[: -(len(t) + 1)]
    return name


def _window(feature: str) -> "int | None":
    '''The trailing number of a feature name: mom20 -> 20, log_adv60 -> 60, beta -> None.'''
    m = re.search(r'(\d+)$', feature)
    return int(m.group(1)) if m else None


def prior_for(name: str) -> "Prior | None":
    '''
    The prior of a signal column, or None if we have none (then stage 05 rejects it unless told otherwise).
    Examples:
        mom5_zscore_tanh   -> reversal, sign -1       mom60_zscore_tanh -> momentum, sign +1
        vol60_zscore_clip  -> low_volatility, -1      log_adv20_zscore_tanh -> illiquidity, -1
        cheat_h20_ic030    -> cheat, +1               placebo_007 -> placebo, +1
        size_rank          -> None
    '''
    if name.startswith('cheat_'):
        return PRIORS['cheat']
    if name.startswith('placebo_'):
        return PRIORS['placebo']

    feature = strip_transform(name)
    window = _window(feature)
    if feature.startswith('mom') and window is not None:
        return PRIORS['reversal'] if window <= SHORT_TERM_MAX_DAYS else PRIORS['momentum']
    if feature.startswith('vol') and window is not None:
        return PRIORS['low_volatility']
    if feature.startswith('beta'):
        return PRIORS['low_beta']
    if feature.startswith('log_adv'):
        return PRIORS['illiquidity']
    return None


def priors_table(names: list) -> dict:
    '''{name: {"family", "sign", "rationale", "reference"} or None} for a list of columns. JSON-safe.'''
    out = {}
    for n in names:
        p = prior_for(n)
        out[n] = None if p is None else {'family': p.family, 'sign': p.sign,
                                         'rationale': p.rationale, 'reference': p.reference}
    return out