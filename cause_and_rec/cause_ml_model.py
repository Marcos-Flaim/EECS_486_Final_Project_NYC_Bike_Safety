
# %%
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error 
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier


# %%
#DOWNLOAD DATA AND MERGE NODE ID WITH THE CRASH DATA 

raw_crash_data = pd.read_csv("crash_data.csv", low_memory=False)
node_map = pd.read_csv("crash_to_node_map.csv")
ranked_list = pd.read_csv("all_intersections_ranked.csv")

#clean headers
raw_crash_data.columns = raw_crash_data.columns.str.strip()
node_map.columns = node_map.columns.str.strip()
ranked_list.columns = ranked_list.columns.str.strip()

#add nodeid to raw crash data

raw_crash_data["COLLISION_ID"] = pd.to_numeric(raw_crash_data["COLLISION_ID"], errors="coerce")
node_map["COLLISION_ID"] = pd.to_numeric(node_map["COLLISION_ID"], errors="coerce")

raw_crash_data = raw_crash_data.dropna(subset=["COLLISION_ID"])
node_map = node_map.dropna(subset=["COLLISION_ID"])

raw_crash_data["COLLISION_ID"] = raw_crash_data["COLLISION_ID"].astype("int64")
node_map["COLLISION_ID"] = node_map["COLLISION_ID"].astype("int64")

#merge crash rows to NODEID
crash_data = raw_crash_data.merge(node_map[["COLLISION_ID", "NODEID"]], on="COLLISION_ID", how="inner")
#remove duplicate columns
crash_data = crash_data.loc[:, ~crash_data.columns.duplicated()]
print("Merged rows:", len(crash_data))
print("Unique intersections:", crash_data["NODEID"].nunique())



# %%
#CREATE FEATURES 

#convert crash date to a datetime, 
# and make some booleans for pedestrian injury etc. 
#boolean about rush hour, weekend, and night

crash_data["CRASH DATE"] = pd.to_datetime(crash_data["CRASH DATE"], errors="coerce")
crash_data["CRASH TIME"] = pd.to_datetime(crash_data["CRASH TIME"], format="%H:%M", errors="coerce")

crash_data["hour"] = crash_data["CRASH TIME"].dt.hour
crash_data["month"] = crash_data["CRASH DATE"].dt.month
crash_data["year"] = crash_data["CRASH DATE"].dt.year
crash_data["dayofweek"] = crash_data["CRASH DATE"].dt.dayofweek

crash_data["is_night"] = ((crash_data["hour"] >= 20) | (crash_data["hour"] <= 5)).astype(int)
crash_data["is_rush_hour"] = crash_data["hour"].isin([7, 8, 9, 16, 17, 18]).astype(int)
crash_data["is_weekend"] = (crash_data["dayofweek"] >= 5).astype(int)

crash_data["is_pedestrian_injury"] = (crash_data["NUMBER OF PEDESTRIANS INJURED"] > 0).astype(int)
crash_data["is_pedestrian_death"] = (crash_data["NUMBER OF PEDESTRIANS KILLED"] > 0).astype(int)
crash_data["is_bike_injury"] = (crash_data["NUMBER OF CYCLIST INJURED"] > 0).astype(int)
crash_data["is_bike_death"] = (crash_data["NUMBER OF CYCLIST KILLED"] > 0).astype(int)
crash_data["is_motorist_injury"] = (crash_data["NUMBER OF MOTORIST INJURED"] > 0).astype(int)
crash_data["is_fatal"] = (crash_data["NUMBER OF PERSONS KILLED"] > 0).astype(int)

crash_data["is_severe"] = (
    (crash_data["NUMBER OF PERSONS KILLED"] > 0) |
    (crash_data["NUMBER OF PERSONS INJURED"] >= 2)
).astype(int)

# %%
#USE CONTRIBUTING FACTOR VEHICLE 1 BECUASE IT IS LESS NOISEY .
#WE CAN ADD CONTR. FAC. 2-4 LATER TO C IF IT IMPROVES MODEL 

cause_col = "CONTRIBUTING FACTOR VEHICLE 1"
crash_data[cause_col] = crash_data[cause_col].fillna("Unspecified").astype(str).str.strip()

#only keep top causes LESS NOISY 
top_causes = crash_data[cause_col].value_counts().nlargest(5).index.tolist()
print("Top causes:", top_causes)


crash_data["cause_label"] = crash_data[cause_col].apply(
    lambda x: x if x in top_causes else "Other"
)

print("\nCause label counts:")
print(crash_data["cause_label"].value_counts())

# %%
#AGGREGATE FEATURES BY THE INTERSECTION OR NODEIDE
# AND CREATE NUMERIC FEATURES /RATIOS BC IT WORLS BETTER WITH XGB 
model_df = crash_data.groupby("NODEID").agg(
    total_crashes=("COLLISION_ID", "count"),
    total_injured=("NUMBER OF PERSONS INJURED", "sum"),
    total_killed=("NUMBER OF PERSONS KILLED", "sum"),
    ped_injury_count=("is_pedestrian_injury", "sum"),
    ped_death_count=("is_pedestrian_death", "sum"),
    bike_injury_count=("is_bike_injury", "sum"),
    bike_death_count=("is_bike_death", "sum"),
    night_crash_count=("is_night", "sum"),
    rush_crash_count=("is_rush_hour", "sum"),
    weekend_crash_count=("is_weekend", "sum"),
    fatal_crash_count=("is_fatal", "sum"),
    avg_injured=("NUMBER OF PERSONS INJURED", "mean")
).reset_index()

#create ratios (works better w xgb)

eps = 1e-6
model_df["night_ratio"] = model_df["night_crash_count"] / (model_df["total_crashes"] + eps)
model_df["rush_ratio"] = model_df["rush_crash_count"] / (model_df["total_crashes"] + eps)
model_df["weekend_ratio"] = model_df["weekend_crash_count"] / (model_df["total_crashes"] + eps)
model_df["ped_injury_ratio"] = model_df["ped_injury_count"] / (model_df["total_crashes"] + eps)
model_df["bike_injury_ratio"] = model_df["bike_injury_count"] / (model_df["total_crashes"] + eps)
model_df["fatal_ratio"] = model_df["fatal_crash_count"] / (model_df["total_crashes"] + eps)

# simple presence flags
model_df["has_ped_crashes"] = (model_df["ped_injury_count"] > 0).astype(int)
model_df["has_bike_crashes"] = (model_df["bike_injury_count"] > 0).astype(int)
model_df["has_night_crashes"] = (model_df["night_crash_count"] > 0).astype(int)
model_df["has_fatal_crashes"] = (model_df["fatal_crash_count"] > 0).astype(int)


# %%
#CREATE 'Y' OR LABELS FOR THE MODEL WITH NODEID AND THE CAUSE

def most_common_label(x):
    mode_vals = x.mode()
    if len(mode_vals) == 0:
        return "Other"
    return mode_vals.iloc[0]

labels_df = crash_data.groupby("NODEID")["cause_label"].apply(most_common_label).reset_index()
labels_df.columns = ["NODEID", "primary_cause"]

# merge features + labels
model_df = model_df.merge(labels_df, on="NODEID", how="inner")

print("\nFinal modeldf:", model_df.shape)
print(model_df.head())

# %%
feature_cols = [
    "total_crashes",
    "total_injured",
    "total_killed",
    "ped_injury_count",
    "ped_death_count",
    "bike_injury_count",
    "bike_death_count",
    "night_crash_count",
    "rush_crash_count",
    "weekend_crash_count",
    "fatal_crash_count",
    "avg_injured",
    "night_ratio",
    "rush_ratio",
    "weekend_ratio",
    "ped_injury_ratio",
    "bike_injury_ratio",
    "fatal_ratio",
    "has_ped_crashes",
    "has_bike_crashes",
    "has_night_crashes",
    "has_fatal_crashes",
]

X = model_df[feature_cols].fillna(0)
y_text = model_df["primary_cause"]

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_text)

X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
    X, y, model_df.index,
    test_size=0.2,
    random_state=42,
    stratify=y
)
# %%
