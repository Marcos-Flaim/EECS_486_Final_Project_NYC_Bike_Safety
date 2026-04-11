import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point
import time
import fiona
from pathlib import Path
import os
import requests
import json

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

# Instead of loading CSV file in chunks, use
# requests.get to pull data from API

# API → dataframe chunk → clean → append/save → discard

# Load in crash data from API in chunks, clean, and convert to GeoDataFrame

limit = 100000 # Adjust as needed
offset = 0
max_rows = 2500000

gdf_list = []

while offset < max_rows:
    url = f"https://data.cityofnewyork.us/resource/h9gi-nx95.json?$limit={limit}&$offset={offset}"
    response = requests.get(url)
    
    if response.status_code != 200:
        print(f"Error fetching data: {response.status_code}")
        break
    
    data_chunk = response.json()
    
    if not data_chunk:
        print("No more data to fetch.")
        break
    
    df_chunk = pd.DataFrame(data_chunk)
    
    # Clean and convert to GeoDataFrame
    df_chunk = df_chunk.dropna(subset=['latitude','longitude'])

    # Convert coords to float
    df_chunk['latitude'] = df_chunk['latitude'].astype(float)
    df_chunk['longitude'] = df_chunk['longitude'].astype(float)

    # Convert numeric cols
    numeric_cols = [
        'number_of_persons_killed',
        'number_of_persons_injured'
    ]

    for col in numeric_cols:
        if col in df_chunk.columns:
            df_chunk[col] = pd.to_numeric(df_chunk[col], errors='coerce')

    df_chunk.columns = (
        df_chunk.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    df_chunk = df_chunk[
        (df_chunk['latitude'] != 0) &
        (df_chunk['longitude'] != 0)
    ]

    # Filter to NYC area
    df_chunk = df_chunk[
        (df_chunk['latitude'].between(40.4, 41)) &
        (df_chunk['longitude'].between(-74.3, -73.6))
    ]
    
    geometry = [
        Point(xy) for xy in zip(
            df_chunk['longitude'],
            df_chunk['latitude']
        )
    ]
    
    gdf_chunk = gpd.GeoDataFrame(
        df_chunk,
        geometry=geometry,
        crs="EPSG:4326"
    )
    
    gdf_list.append(gdf_chunk)

    print(f"Fetched {len(df_chunk)} records. Total so far: {len(pd.concat(gdf_list))}")
    
    offset += limit
    time.sleep(1)  # Be respectful of API rate limits

crash_gdf = pd.concat(gdf_list, ignore_index=True)

crash_gdf = crash_gdf.to_crs(2263)

# Get path to lion.gdb (***change if necessary)
# Should be in same folder as this script, but within a 
# "lion" subfolder

# Path to current directory
current_dir = Path(__file__).parent
# Path to lion.gdb
gdb_path = current_dir / "lion" / "lion.gdb"

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
snapped["crash_date"] = pd.to_datetime(snapped["crash_date"])

# Compute age in yrs
today = pd.Timestamp.today()
snapped["age_years"] = (today - snapped["crash_date"]).dt.days / 365.25

# Exp decay parameter
lambda_ = 0.3 #TODO:REVISE/TUNE--------------

snapped["recency_weight"] = np.exp(-lambda_ * snapped["age_years"])

#Base severity per crash
snapped["base_severity"] = (
    10 * snapped["number_of_persons_killed"].fillna(0) +
    3 * snapped["number_of_persons_injured"].fillna(0) +
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
        total_killed=("number_of_persons_killed", "sum"),
        total_injured=("number_of_persons_injured", "sum"),
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

# sam added: need a centralized location for all the data since work is compartmentalized
if not Path('../data').exists():
    os.mkdir('../data')

# Save first 500 rows to CSV
results.head(500).to_csv("../data/intersection_rankings.csv", index=False)


#uma added: want a crash_to_node_map so we usee same intersection break down in model 
snapped[['collision_id', 'NODEID', 'snap_dist']].to_csv("../data/crash_to_node_map.csv", index=False)
# TESTS-------------------------
# Snapping quality
print(snapped["snap_dist"].describe())
# Crash count distribution
print(node_counts["crash_count"].describe())


# Print num crashes for each year for NODEID 36094 (example intersection)
example_nodeid = 36094
example_snapped = snapped[snapped["NODEID"] == example_nodeid]
print(example_snapped.groupby(example_snapped["crash_date"].dt.year).size())


