import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point
import time
import fiona

# 10am, 2/27: Using LION node data instead of street centerline
# shapefile for intersection-specific data
# LION node layer contains built-in intersection points, which can
# be used to analyze crash data at intersections more accurately
# than using street centerlines.

# GOAL: cluster on unique crash records rather than 
# collapsing data into unique lat/long coordinates
# in order to weigh severity, frequency, etc of incidents
# and ID hot spots 

# Currently have list of intersections with assigned crash counts,
# need to:
# (1) revise if we want to drop non-intersecctions (ex. Van Wyck expwy) 
# (2) rank by severity (formula below is starting point)

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

# Create GeoDataFrame with each crash instance
geometry = [
    Point(xy) for xy in zip(
        df['LONGITUDE'],
        df['LATITUDE']
    )
]

crash_gdf = gpd.GeoDataFrame(
    df,
    geometry=geometry,
    crs="EPSG:4326"
)

crash_gdf = crash_gdf.to_crs(2263)

<<<<<<< HEAD
# List layers inside lion.gdb
gdb_path = "lion/lion.gdb"
layers = fiona.listlayers(gdb_path)
print("Layers in lion.gdb:", layers)
=======
# Get path to lion.gdb
gdb_path = "lion/lion.gdb"
>>>>>>> 01b98d021ed5c1dc67ad3024249fd5bada921fd8

# Load the node layer
nodes = gpd.read_file(gdb_path, layer="node")
nodes = nodes.to_crs(2263)
# TODO: Filter out non-intersection nodes

# Load centerline layer
lion = gpd.read_file(gdb_path, layer="lion")
lion = lion.to_crs(2263)

# ----PROCESS OVERVIEW:
# Crash instances -> snap to official LION NODEID -> filter
# -> count -> label intersections

# Snap every crash to nearest NODEID (intersection point)
snapped = gpd.sjoin_nearest(
    crash_gdf,
    nodes[["NODEID", "geometry"]],
    how="left",
    distance_col="snap_dist"
)

# Filter to crashes within ~50ft of node 
# (ensures we're capturing intersection crashes, not mid-block)
snapped = snapped[snapped["snap_dist"] <= 50]

# Group crash count by NODEID
node_counts = (
    snapped.groupby("NODEID")
    .size()
    .reset_index(name="crash_count")
    .sort_values("crash_count", ascending=False)
)

# Extract street names for each NODEID
from_nodes = lion[["NodeIDFrom", "Street"]].rename(
    columns={"NodeIDFrom": "NODEID"}
)

to_nodes = lion[["NodeIDTo", "Street"]].rename(
    columns={"NodeIDTo": "NODEID"}
)

node_streets = pd.concat([from_nodes, to_nodes])

# Drop duplicates to get unique street names per NODEID
node_streets = node_streets.drop_duplicates()

# Group street names by NODEID
intersection_names = (
    node_streets
    .groupby("NODEID")["Street"]
    .unique()
    .reset_index()
)

intersection_names["intersection_name"] = (
    intersection_names["Street"]
    .apply(lambda x: " & ".join(sorted(x)))
)

# Merge crash counts with intersection names
node_counts["NODEID"] = node_counts["NODEID"].astype(int)
intersection_names["NODEID"] = intersection_names["NODEID"].astype(int)
results = node_counts.merge(
    intersection_names[["NODEID", "intersection_name"]],
    on="NODEID",
    how="left"
)

# Extract latitude and longitude for each intersection -------
# (nyclion data tracks coordinates via NY State Plane (EPSG:2263), 
# we can convert back to universal lat/long for mapping)

# Get unique State Plane coordinates per NODEID
node_coords = nodes[["NODEID", "geometry"]].drop_duplicates()
node_coords["latitude"] = node_coords.geometry.y
node_coords["longitude"] = node_coords.geometry.x

# Group coordinates by NODEID
intersection_coords = (
    node_coords
    .groupby("NODEID")[["latitude", "longitude"]]
    .first()
    .reset_index()
)

# Convert latitude and longitude back to WGS84 for output
intersection_coords_gdf = gpd.GeoDataFrame(
    intersection_coords,
    geometry=gpd.points_from_xy(
        intersection_coords["longitude"],
        intersection_coords["latitude"]
    ),
    crs=2263
)

intersection_coords_gdf = intersection_coords_gdf.to_crs(4326)
intersection_coords = pd.DataFrame({
    "NODEID": intersection_coords_gdf["NODEID"],
    "latitude": intersection_coords_gdf.geometry.y,
    "longitude": intersection_coords_gdf.geometry.x
})

# Merge with intersection names and crash counts
results = results.merge(
    intersection_coords,
    on="NODEID",
    how="left"
)

# ----Heuristic formula to rank intersection severity:
# Severity weighting - num killed/injured, crash count
# Recency weighting - more recent crashes weighted higher

# 1) Calculate weighted score (for each crash)
snapped["CRASH DATE"] = pd.to_datetime(snapped["CRASH DATE"])

# Compute age in yrs
today = pd.Timestamp.today()
snapped["age_years"] = (today - snapped["CRASH DATE"]).dt.days / 365.25

# Exp decay parameter
lambda_ = 0.3 #TODO:REVISE/TUNE--------------

snapped["recency_weight"] = np.exp(-lambda_ * snapped["age_years"])

#Base severity per crash
snapped["base_severity"] = (
    10 * snapped["NUMBER OF PERSONS KILLED"].fillna(0) +
    3 * snapped["NUMBER OF PERSONS INJURED"].fillna(0) +
    1
)

# Final weighted crash score
snapped["weighted_score"] = (
    snapped["base_severity"] * snapped["recency_weight"]
)

# 2) Create intersections summary
node_summary = (
    snapped.groupby("NODEID")
    .agg(
        crash_count=("NODEID", "size"),
        total_killed=("NUMBER OF PERSONS KILLED", "sum"),
        total_injured=("NUMBER OF PERSONS INJURED", "sum"),
        # Severity score = sum of weighted scores for each crash at that intersection
        severity_score=("weighted_score", "sum")
    )
    .reset_index()
    .sort_values("severity_score", ascending=False)
)

# 3) Add normalized severity score (0-1) for easier comparison ---TODO: REVISE
node_summary["severity_score_norm"] = (
    node_summary["severity_score"] - node_summary["severity_score"].min()
) / (node_summary["severity_score"].max() - node_summary["severity_score"].min()
)

# Merge w intersection names
node_summary["NODEID"] = node_summary["NODEID"].astype(int)
intersection_names["NODEID"] = intersection_names["NODEID"].astype(int)

results = node_summary.merge(
    intersection_names[["NODEID", "intersection_name"]],
    on="NODEID",
    how="left"
)

# Merge w lat/long
results = results.merge(
    intersection_coords,
    on="NODEID",
    how="left"
)

# Print first 10 rows of results
print(results.head(10)[
    ["intersection_name",
    "crash_count",
    "total_killed",
    "total_injured",
    "severity_score",
    "severity_score_norm",
    "latitude",
    "longitude",
    "NODEID"]
])

<<<<<<< HEAD
#uma added: print to "all_intersections_ranked.csv"
results[
    ["intersection_name",
     "crash_count",
     "total_killed",
     "total_injured",
     "severity_score"]].to_csv("all_intersections_ranked.csv", index=False)
=======
# Save first 500 rows to CSV
results.head(500).to_csv("intersection_rankings.csv", index=False)

>>>>>>> 01b98d021ed5c1dc67ad3024249fd5bada921fd8

#uma added: want a crash_to_node_map so we usee same intersection break down in model 
snapped[['COLLISION_ID', 'NODEID', 'snap_dist']].to_csv("crash_to_node_map.csv", index=False)
# TESTS-------------------------
# Snapping quality
print(snapped["snap_dist"].describe())
# Crash count distribution
print(node_counts["crash_count"].describe())


# Print num crashes for each year for NODEID 36094 (example intersection)
example_nodeid = 36094
example_snapped = snapped[snapped["NODEID"] == example_nodeid]
print(example_snapped.groupby(example_snapped["CRASH DATE"].dt.year).size())


