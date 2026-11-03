import math
import time

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import haversine_distances

URL = ('https://gist.githubusercontent.com/jlewis8756/'
       '6b83a54351e91012b9fd541356a347c9/raw/world_cities.csv')
EARTH_RADIUS_KM = 6371.0088

# ---------------------------------------------------------------
# Setup: same data as p4. The radians conversion happens once,
# outside every timer, so each implementation below is timed on
# the distance computation alone.
# ---------------------------------------------------------------
df = pd.read_csv(URL)
big = df[df['population'] > 1_000_000].reset_index(drop=True)
coords = np.radians(big[['lat', 'lng']].to_numpy())
points = [tuple(p) for p in coords]
n = len(points)
print(f"{n} cities >1M -> {n * n:,} pairwise distances\n")

# ---------------------------------------------------------------
# 1) scikit-learn haversine_distances (best of 3 runs)
# ---------------------------------------------------------------
sk_times = []
for _ in range(3):
    t0 = time.perf_counter()
    dist_sklearn = haversine_distances(coords)
    sk_times.append((time.perf_counter() - t0) * 1000)
sk_ms = min(sk_times)

# ---------------------------------------------------------------
# 2) pure Python: nested loops with the math module
# ---------------------------------------------------------------
sin, cos, asin, sqrt = math.sin, math.cos, math.asin, math.sqrt

t0 = time.perf_counter()
dist_pure = [[0.0] * n for _ in range(n)]
for i in range(n):
    lat1, lon1 = points[i]
    row = dist_pure[i]
    for j in range(n):
        lat2, lon2 = points[j]
        a = (sin((lat2 - lat1) / 2) ** 2
             + cos(lat1) * cos(lat2) * sin((lon2 - lon1) / 2) ** 2)
        row[j] = 2 * EARTH_RADIUS_KM * asin(sqrt(1.0 if a > 1.0 else a))
pure_ms = (time.perf_counter() - t0) * 1000

# ---------------------------------------------------------------
# Correctness: both should produce the same matrix
# ---------------------------------------------------------------
dist_pure = np.array(dist_pure)
max_diff = np.abs(dist_sklearn - dist_pure).max()

# ---------------------------------------------------------------
# Results
# ---------------------------------------------------------------
print(f"sklearn      : {sk_ms:9.2f} ms   (best of 3)")
print(f"pure Python  : {pure_ms:9.2f} ms")
print(f"speedup      : {pure_ms / sk_ms:9.1f}x faster with sklearn")
print(f"max abs diff : {max_diff:9.2e} km")
print("\nCUDA (proj2) : skipped - project 2 is not submitted yet")
