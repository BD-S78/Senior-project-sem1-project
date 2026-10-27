import sys
import time
import timeit
import math

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append('./build')

import haversine_library

# data from https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
BBOX = (-74.15, 40.5774, -73.7004, 40.9176)
COLUMNS = ["Start_Lon", "Start_Lat", "End_Lon", "End_Lat",
           "Trip_Pickup_DateTime"]

lon_min, lat_min, lon_max, lat_max = BBOX
monthly = []
for month in range(1, 13):
    path = f"/tmp/tlcdata/yellow_tripdata_2009-{month:02d}.parquet"
    df = pd.read_parquet(path, columns=COLUMNS)
    monthly.append(df.query(
        "(Start_Lon >= @lon_min) and (Start_Lon <= @lon_max) and "
        "(Start_Lat >= @lat_min) and (Start_Lat <= @lat_max) and "
        "(End_Lon >= @lon_min) and (End_Lon <= @lon_max) and "
        "(End_Lat >= @lat_min) and (End_Lat <= @lat_max)"))

taxi = pd.concat(monthly, ignore_index=True)
del monthly
print("filtered trips:", len(taxi))

weekend = (pd.to_datetime(taxi["Trip_Pickup_DateTime"]).dt.dayofweek >= 5).to_numpy()

x1 = taxi["Start_Lon"].to_numpy(dtype=np.float64)
y1 = taxi["Start_Lat"].to_numpy(dtype=np.float64)
x2 = taxi["End_Lon"].to_numpy(dtype=np.float64)
y2 = taxi["End_Lat"].to_numpy(dtype=np.float64)
del taxi

size = len(x1)
dist = np.zeros(size)

# warm up CUDA, then time the whole call from Python
haversine_library.haversine_distance(size, x1, y1, x2, y2, dist)
gpu_time = timeit.timeit(
    lambda: haversine_library.haversine_distance(size, x1, y1, x2, y2, dist),
    number=3) / 3
times = haversine_library.last_timings()


def haversine_python(x1, y1, x2, y2):
    R = 6371.0
    d2r = math.pi / 180.0
    out = np.empty(len(x1))
    step = 1_000_000
    for start in range(0, len(x1), step):
        end = min(start + step, len(x1))
        for i, (lo1, la1, lo2, la2) in enumerate(zip(
                x1[start:end].tolist(), y1[start:end].tolist(),
                x2[start:end].tolist(), y2[start:end].tolist())):
            lat1 = la1 * d2r
            lat2 = la2 * d2r
            dlat = (la2 - la1) * d2r
            dlon = (lo2 - lo1) * d2r
            a = (math.sin(dlat / 2) ** 2
                 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2)
            out[start + i] = 2 * R * math.asin(math.sqrt(min(1.0, a)))
    return out


t0 = time.perf_counter()
dist_py = haversine_python(x1, y1, x2, y2)
py_time = time.perf_counter() - t0
assert np.allclose(dist, dist_py, atol=1e-6)
del dist_py

# times = [GPU malloc, CPU->GPU, kernel, GPU->CPU] in milliseconds
print(f"GPU malloc: {times[0]:.3f} ms")
print(f"CPU->GPU:   {times[1]:.3f} ms")
print(f"kernel:     {times[2]:.3f} ms")
print(f"GPU->CPU:   {times[3]:.3f} ms")
print(f"GPU end-to-end: {gpu_time * 1000:.3f} ms")
print(f"Python loop:    {py_time * 1000:.3f} ms")
print(f"kernel speedup:   {py_time / (times[2] / 1000):.1f}x")
print(f"end-to-end speedup: {py_time / gpu_time:.1f}x")

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, values, label in zip(
        axes.flat, [x1, y1, x2, y2, dist],
        ["Start_Lon", "Start_Lat", "End_Lon", "End_Lat", "Distance (km)"]):
    ax.hist(values, bins=100)
    ax.set_title(label)
axes.flat[-1].axis("off")
plt.tight_layout()
plt.savefig("histograms.png", dpi=120)

summary = (pd.DataFrame({"weekend": weekend, "distance_km": dist})
           .groupby("weekend")
           .agg(trips=("distance_km", "size"),
                avg_distance_km=("distance_km", "mean")))
summary.rename(index={False: "weekday", True: "weekend"}, inplace=True)
print(summary)
