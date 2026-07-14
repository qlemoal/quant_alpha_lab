from polars import col as c

##  NOT READY

def add_mkt_logret(lf):
    market_logret = (
        lf.group_by('date')
        .agg(c('logret').mean())
        .alias('mkt_logret')
    )
    lf = lf.join(market_logret)

def add_beta(lf):
