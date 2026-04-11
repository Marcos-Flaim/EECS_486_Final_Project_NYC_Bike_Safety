import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
import time

# Load Excel file
df = pd.read_excel("crash_test_data.xlsx")

df = df.dropna(subset=['LATITUDE','LONGITUDE'])

df = df[
    (df['LATITUDE'] != 0) &
    (df['LONGITUDE'] != 0)
]

geometry = [Point(xy) for xy in zip(df['LONGITUDE'], df['LATITUDE'])]
crash_gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

streets = gpd.read_file("nyc_centerline.shp")

# Reproject to projected CRS
crash_gdf = crash_gdf.to_crs(2263)
streets = streets.to_crs(2263)

start = time.time()

joined = gpd.sjoin_nearest(
    crash_gdf,
    streets,
    how="left",
    distance_col="distance"
)

end = time.time()
print(f"Spatial join completed in {end - start:.2f} seconds")

cols_to_view = [
    'CRASH DATE',
    'LATITUDE',
    'LONGITUDE',
    'stname_lab',
    'physicalid',
    'distance'
]

#print(joined[cols_to_view].head(10))
#print(joined['distance'].describe())
#print(joined[joined['distance'] > 100][
#    ['LATITUDE','LONGITUDE','stname_lab','distance']
#])

row = joined.iloc[0]

print("Street:", row['stname_lab'])
print("Distance (ft):", row['distance'])
print("Crash Date:", row['CRASH DATE'])