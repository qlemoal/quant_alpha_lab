


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