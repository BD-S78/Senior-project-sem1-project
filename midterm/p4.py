import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import haversine_distances
from sklearn.cluster import AgglomerativeClustering


URL = ('https://gist.githubusercontent.com/jlewis8756/'
       '6b83a54351e91012b9fd541356a347c9/raw/world_cities.csv')
print(f"Reading {URL} ...")
df = pd.read_csv(URL)
big = df[df['population'] > 1_000_000].copy().reset_index(drop=True)
print(f"Step 1: {len(big)} cities with population > 1M")

coords = np.radians(big[['lat', 'lng']].to_numpy())
dist_km = haversine_distances(coords) * 6371.0088
print(f"Step 2: {dist_km.shape} distance matrix, max {dist_km.max():.0f} km")

model = AgglomerativeClustering(
    n_clusters=10, linkage='average', metric='precomputed')
big['cluster'] = model.fit_predict(dist_km)
sizes = sorted(np.bincount(big['cluster']).tolist(), reverse=True)
print(f"Step 3: 10 clusters, sizes: {sizes}")

plt.figure(figsize=(14, 7))
plt.scatter(big['lng'], big['lat'], c=big['cluster'], cmap='tab10', s=18)
plt.title('AgglomerativeClustering of world cities >1M (k=10, average linkage)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.tight_layout()
plt.savefig('p4_clusters.png', dpi=150)
print('Saved p4_clusters.png')