import os
import time

import pandas as pd

path = '/tmp/tlcdata/yellow_tripdata_2009-01.parquet'
csv_path = '/tmp/converted.csv'  # /tmp/tlcdata is read-only on the GPU server

t0 = time.perf_counter()
df = pd.read_parquet(path)
parquet_ms = (time.perf_counter() - t0) * 1000

df.to_csv(csv_path, index=False)

t0 = time.perf_counter()
df_csv = pd.read_csv(csv_path)
csv_ms = (time.perf_counter() - t0) * 1000

parquet_mb = os.path.getsize(path) / 1e6
csv_mb = os.path.getsize(csv_path) / 1e6

print(f"file:    {os.path.basename(path)}")
print(f"rows:    {len(df):,}")
print(f"parquet: {parquet_mb:8.1f} MB  load {parquet_ms:8.1f} ms")
print(f"csv:     {csv_mb:8.1f} MB  load {csv_ms:8.1f} ms")
print(f"csv is {csv_mb / parquet_mb:.1f}x larger and {csv_ms / parquet_ms:.1f}x slower")