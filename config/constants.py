


TRADING_DAYS = 252

ROLLING_WINDOWS = {
    'tiny' : 5,
    'short' : 20,
    'medium' : 60,
    'long' : 252,
}


LONG_THRESHOLD = 0.8
SHORT_THRESHOLD = -0.8

#  To find the right embargo window, run embargo_selection.py on the right time series, 
#       i.e., if 5-day forward returns are the target, compute ACF of 5-day aggregated returns
EMBARGO_WINDOW = 16

#  HOLDOUT: the last HOLDOUT_DATES labelled dates are never used to select, tune or fit (stages 05, 06).
#       Why 3 years: the standard error of an annual Sharpe ratio is about sqrt(1 / years).
#       200 dates (0.8 year) give a 95% interval of about +-2.2: it cannot tell a good strategy from a bad one.
#       756 dates (3 years) give about +-1.1. More years = tighter interval, but fewer years to search on.
#       Change it ONCE, before the first real run. Changing it after you saw a holdout result is snooping.
HOLDOUT_DATES = 756