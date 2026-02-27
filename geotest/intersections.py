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

# List layers inside lion.gdb
gdb_path = "nyclion/lion/lion.gdb"
layers = fiona.listlayers(gdb_path)
print("Layers in lion.gdb:", layers)

# Load the node layer
nodes = gpd.read_file(gdb_path, layer="node")
nodes = nodes.to_crs(2263)
# TODO: Filter out non-intersection nodes
print("Node layer loaded with", len(nodes), "features")
print(nodes.columns)

# Load centerline layer
lion = gpd.read_file(gdb_path, layer="lion")
lion = lion.to_crs(2263)
print("Lion layer loaded with", len(lion), "features")
print(lion.columns)

# Confirm Nodeto and Nodefrom exist
print([col for col in lion.columns if "Node" in col])

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

print(snapped["snap_dist"].describe())

# Count crashes per NODEID
node_counts = (
    snapped.groupby("NODEID")
    .size()
    .reset_index(name="crash_count")
    .sort_values("crash_count", ascending=False)
)
print(node_counts.head(10))

# Extract street names per NODEID
from_nodes = lion[["NodeIDFrom", "Street"]].rename(
    columns={"NodeIDFrom": "NODEID"}
)

to_nodes = lion[["NodeIDTo", "Street"]].rename(
    columns={"NodeIDTo": "NODEID"}
)

node_streets = pd.concat([from_nodes, to_nodes])

# Drop duplicates to get unique street names per NODEID
node_streets = node_streets.drop_duplicates()

# Group street names per NODEID
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

print(results.head(10))
print(results.head(10)[["intersection_name", "crash_count"]])

# IN PROGRESS: Add heuristic formula to rank intersection severity
# Severity weighting - num killed/injured, crash count
# Recency weighting - more recent crashes weighted higher

snapped["CRASH DATE"] = pd.to_datetime(snapped["CRASH DATE"])

# Compute age in yrs
today = pd.Timestamp.today()
snapped["age_years"] = (today - snapped["CRASH DATE"]).dt.days / 365.25

# Exp decay parameter
lambda_ = 0.3 # REVISE/TUNE--------------

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

# Create summary by weighted score
node_summary = (
    snapped.groupby("NODEID")
    .agg(
        crash_count=("NODEID", "size"),
        total_killed=("NUMBER OF PERSONS KILLED", "sum"),
        total_injured=("NUMBER OF PERSONS INJURED", "sum"),
        severity_score=("weighted_score", "sum")
    )
    .reset_index()
    .sort_values("severity_score", ascending=False)
)

print(node_summary.head(10))

# Merge w intersection names
node_summary["NODEID"] = node_summary["NODEID"].astype(int)
intersection_names["NODEID"] = intersection_names["NODEID"].astype(int)

results = node_summary.merge(
    intersection_names[["NODEID", "intersection_name"]],
    on="NODEID",
    how="left"
)

print(results.head(10)[
    ["intersection_name",
     "crash_count",
     "total_killed",
     "total_injured",
     "severity_score"]
])


# TESTS-------------------------
# Snapping quality
print(snapped["snap_dist"].describe())
# Crash count distribution
print(node_counts["crash_count"].describe())




