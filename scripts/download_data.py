from pathlib import Path

import pandas as pd
import yfinance as yf
from tqdm import tqdm
import requests

RAW_DIR = Path('data/raw/prices')
RAW_DIR.mkdir(parents=True, exist_ok=True)

print('Downloading SP500 tickers one by one, using wikipedia to stay up-to-date')

wiki_website = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'

# Need an agent to be accepted by wikipedia 
headers = {
    "User-Agent": "Mozilla/5.0"
}

# The requests tell us what is the status of our connection: 403: not allowed, 200: accepted.
r = requests.get(wiki_website, headers=headers)
r.status_code

table = pd.read_html(r.text)[0]

tickers = table['Symbol'].tolist()

print(f'{len(tickers)} tickers found')

for ticker in tqdm(tickers[:]):

    try:
        df = yf.download(
            ticker, 
            start = '2000-01-01',
            auto_adjust = False,
            progress=False
        )

        if len(df) > 1:
            df.to_csv(RAW_DIR / f'{ticker}.csv')

    except Exception:
        print(f'failed: {ticker}')

print( 'Done')