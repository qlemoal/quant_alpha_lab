from pathlib import Path

import duckdb

RAW_DIR = Path('data/raw/prices')
OUT_DIR = Path('data/processed')

OUT_DIR.parent.mkdir(exist_ok = True)

print(f'Aggregating all prices from {RAW_DIR}')

con = duckdb.connect()
con.execute(f'''
CREATE OR REPLACE TABLE prices AS

SELECT
            filename, Date, Open, High, Low, Close, Volume
FROM            
            read_csv_auto('{RAW_DIR}/*.csv', filename=True)
''')

print('Done')