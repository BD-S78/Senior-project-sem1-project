import pandas as pd

data = {
    'kind': ['cat', 'dog', 'cat', 'dog'],
    'height': [9.1, 6.0, 9.5, 34.0],
    'weight': [7.9, 7.5, 9.9, 198.0]
}

df = pd.DataFrame(data)
print("pandas implementation:")
print(df.groupby('kind')['height'].agg(['min', 'max']))


#bonus cuDF solution
try:
    import cudf
    gdf = cudf.DataFrame(data)
    print("\ncuDF implementation:")
    print(gdf.groupby('kind')['height'].agg(['min', 'max']))
except Exception as e:
    print(f"\ncuDF implementation skipped: ({type(e).__name__})")