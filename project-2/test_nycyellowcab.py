import sys
import time
import timeit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append('./build')

import haversine_library

#code from: https://github.com/rapidsai/cuspatial/blob/724d170a2105441a3533b5eaf9ee82ddcfc49be0/notebooks/nyc_taxi_years_correlation.ipynb
#data from https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

# Step 4: read 12 months, filter to NYC bounding box, combine into one dataframe
bbox_query = (
    "-74.15 <= Start_Lon <= -73.7004 and "
    "40.5774 <= Start_Lat <= 40.9176 and "
    "-74.15 <= End_Lon <= -73.7004 and "
    "40.5774 <= End_Lat <= 40.9176"
)

monthly = []
for month in range(1, 13):
    taxi = pd.read_parquet(f"/tmp/tlcdata/yellow_tripdata_2009-{month:02d}.parquet")
    monthly.append(taxi.query(bbox_query))

taxi = pd.concat(monthly, ignore_index=True)
print(f"total filtered trips: {len(taxi)}")

x1=taxi['Start_Lon'].to_numpy(dtype=np.float64)
y1=taxi['Start_Lat'].to_numpy(dtype=np.float64)
x2=taxi['End_Lon'].to_numpy(dtype=np.float64)
y2=taxi['End_Lat'].to_numpy(dtype=np.float64)
size=len(x1)
dist=np.zeros(size)

t0=time.time()
haversine_library.haversine_distance(size,x1,y1,x2,y2,dist)
t1=time.time()
print(f"GPU end-to-end: {(t1-t0)*1000:.3f} ms")

# Step 5: Python haversine for comparison
#reference: same formula as the CUDA kernel
def haversine_py(x1, y1, x2, y2):
    R=6371.0
    d2r=np.pi/180.0
    lat1=y1*d2r
    lat2=y2*d2r
    dlat=(y2-y1)*d2r
    dlon=(x2-x1)*d2r
    a=np.sin(dlat/2)**2+np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return R*2.0*np.arctan2(np.sqrt(a),np.sqrt(1.0-a))

dist_py=haversine_py(x1,y1,x2,y2)
assert np.allclose(dist,dist_py,atol=1e-6)

py_time=min(timeit.repeat(lambda: haversine_py(x1,y1,x2,y2),number=1,repeat=3))
print(f"Python numpy: {py_time*1000:.3f} ms")
print(f"GPU end-to-end speedup: {py_time/(t1-t0):.2f}x")

# Step 5 bonus: histograms
fig,axes=plt.subplots(2,3,figsize=(15,8))
for idx,(arr,lbl) in enumerate(zip(
    [x1,y1,x2,y2,dist],
    ['Start_Lon','Start_Lat','End_Lon','End_Lat','Distance (km)']
)):
    ax=axes[idx//3][idx%3]
    ax.hist(arr,bins=100)
    ax.set_title(lbl)
axes[1][2].set_visible(False)
plt.tight_layout()
plt.savefig("histograms.png",dpi=120)
plt.show()

# Step 6 bonus: weekday vs weekend trip count and avg distance
datetime_col=next((c for c in ['pickup_datetime','Trip_Pickup_DateTime','tpep_pickup_datetime'] if c in taxi.columns),None)
if datetime_col:
    taxi['pickup_dt']=pd.to_datetime(taxi[datetime_col])
    taxi['is_weekend']=taxi['pickup_dt'].dt.dayofweek>=5
    taxi['distance_km']=dist
    summary=taxi.groupby('is_weekend')['distance_km'].agg(num_trips='count',avg_distance_km='mean')
    summary.index=summary.index.map({False:'Weekday',True:'Weekend'})
    print(summary)
