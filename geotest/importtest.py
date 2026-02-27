import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
import time

# Load CSV file in chunks
df_iter = pd.read_csv(
    "crash_data.csv",
    chunksize=200000,
    dtype={"ZIP CODE": str},
    low_memory=False
)

clean_chunks = []

for chunk in df_iter:

    chunk = chunk.dropna(subset=['LATITUDE','LONGITUDE'])

    chunk = chunk[
        (chunk['LATITUDE'] != 0) &
        (chunk['LONGITUDE'] != 0)
    ]

    clean_chunks.append(chunk)

df = pd.concat(clean_chunks)

# Filter to NYC area
df = df[
    (df['LATITUDE'].between(40.4, 41.0)) &
    (df['LONGITUDE'].between(-74.3, -73.6))
]

unique_coords = df[['LATITUDE','LONGITUDE']].drop_duplicates()

geometry = [
    Point(xy) for xy in zip(
        unique_coords['LONGITUDE'],
        unique_coords['LATITUDE']
    )
]

# Create GeoDataFrame with unique coordinates
crash_gdf_unique = gpd.GeoDataFrame(
    unique_coords,
    geometry=geometry,
    crs="EPSG:4326"
)

# Load street centerline shapefile
streets = gpd.read_file("nyc_centerline.shp")

# Reproject to projected CRS
crash_gdf_unique = crash_gdf_unique.to_crs(2263)
streets = streets.to_crs(2263)

# Perform spatial join with unique crash points, time it
start = time.time()

joined_unique = gpd.sjoin_nearest(
    crash_gdf_unique,
    streets,
    how="left",
    distance_col = "distance"
)

# Merge the joined results back to the original DataFrame
df = df.merge(
    joined_unique[['LATITUDE','LONGITUDE','stname_lab','distance']],
    on=['LATITUDE','LONGITUDE'],
    how='left'
)

print("Join runtime:", time.time() - start)
print("Unique coords: ", len(unique_coords))
print("Total crashes: ", len(df))
print(df[['LATITUDE','LONGITUDE','stname_lab','distance']].sample(10))
print(df['distance'].describe())
street_counts = df.groupby("stname_lab").size().sort_values(ascending=False)
print(street_counts.head(10))
print(df['distance'].quantile([0.5, 0.75, 0.9, 0.99]))