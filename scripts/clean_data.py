from pathlib import Path

import duckdb

RAW_DIR = Path('data/raw/prices')
OUT_DIR = Path('data/processed')

OUT_DIR.parent.mkdir(exist_ok = True)

print(f'Aggregating all prices from {RAW_DIR}')


#  Agggregate all the CSVs into one prices table
con = duckdb.connect()
con.execute(f'''
CREATE OR REPLACE TABLE prices AS
SELECT
    filename, Date, Open, High, Low, Close, Volume
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
    CAST(Close AS DOUBLE) AS close,
    CAST(Volume AS BIGINT) AS volume
FROM 
    prices
WHERE
    Close > 0
AND 
    Volume >= 0
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
TO '{OUT_DIR}/prices.parquet'
(FORMAT PARQUET);
''')

print("Saved.")