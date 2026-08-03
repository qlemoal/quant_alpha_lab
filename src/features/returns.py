from polars import col as c






def add_log_returns(lf):
    return lf.sort(['ticker', 'date']).with_columns(
                logret = c('close').log().diff().over('ticker')
            )



def add_fwd_returns(lf):
    return lf.sort(['ticker', 'date']).with_columns(
                fwdret = c('logret').shift(-1).over('ticker').alias('fwdret')
            )