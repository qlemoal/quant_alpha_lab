from pathlib import Path
from datetime import date

import duckdb

RAW_DIR = Path('data/raw/ohlcv')
RUN_DATE = date.today().isoformat()
SIGNAL_TAG = 'default'
OUT_DIR = Path(f'data/processed/ohlcv/{RUN_DATE}_{SIGNAL_TAG}')
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f'Aggregating all prices from {RAW_DIR}')


#  Aggregate all the CSVs into one prices table
con = duckdb.connect()
con.execute(f'''
CREATE OR REPLACE TABLE prices AS
SELECT
    filename, Date, Open, High, Low, "Adj Close", Volume
FROM            
    read_csv_auto('{RAW_DIR}/*.csv', filename=True)
''')

#  Clean the data types, extract the ticker name, lowercase column names, and filter positive prices and volume days
con.execute('''
CREATE OR REPLACE TABLE clean AS
SELECT  
    regexp_extract(filename,'([^/]+)\.csv',1) AS ticker,
    CAST(Date AS DATE) AS date,
    CAST(Open AS DOUBLE) AS open,
    CAST(High AS DOUBLE) AS high,
    CAST(Low AS DOUBLE) AS low,
    CAST("Adj Close" AS DOUBLE) AS close,
    CAST(Volume AS BIGINT) AS volume
FROM 
    prices
WHERE
    "Adj Close" > 0
AND 
    Volume > 0
''')


#  Remove duplicates
con.execute('''
CREATE OR REPLACE TABLE clean AS
SELECT DISTINCT 
    *
FROM 
    clean
''')


#  Sort by ticker and date, and write the final parquet file to OUT_DIR
con.execute(f'''
COPY(
    SELECT 
        *
    FROM 
        clean
    ORDER BY 
        ticker, date
)
TO '{OUT_DIR}/ohlcv.parquet'
(FORMAT PARQUET);
''')

# Save config
import json
config = {
    'run_date': RUN_DATE,
    'signal_tag': SIGNAL_TAG,
    'source': 'yfinance',
    'start_date': '2000-01-01',
    'raw_dir': str(RAW_DIR),
    'output_file': 'ohlcv.parquet',
}
with open(OUT_DIR / 'config.json', 'w') as f:
    json.dump(config, f, indent=2)

print(f"Saved to {OUT_DIR}/ohlcv.parquet")
print(f"Config saved to {OUT_DIR}/config.json")