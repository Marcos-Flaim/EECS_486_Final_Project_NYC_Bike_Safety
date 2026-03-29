
# %%
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd 
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, classification_report, accuracy_score, confusion_matrix, f1_score
from xgboost import XGBClassifier
import plotly.express as px


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
# Before creating cause_label, filter out rows where cause was "Unspecified"
crash_data = crash_data[crash_data[cause_col] != "Unspecified"]

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
model_df["ped_proportion"] = model_df["ped_injury_count"] / (model_df["total_injured"] + eps)
model_df["bike_proportion"] = model_df["bike_injury_count"] / (model_df["total_injured"] + eps)
model_df["severe_ratio"] = model_df["fatal_crash_count"] / (model_df["total_crashes"] + eps)

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
    "ped_proportion",
    "bike_proportion",
    "severe_ratio",
    "has_ped_crashes",
    "has_bike_crashes",
    "has_night_crashes",
    "has_fatal_crashes",
]

# Run the model on top 500 ranked intersections, so filter crash_data to just those NODEIDs first.
# Make a new column for intersection name in crash_data
crash_data["intersection_name"] = (
    crash_data["ON STREET NAME"].str.strip().str.upper()
    + " & " +
    crash_data["CROSS STREET NAME"].str.strip().str.upper()
)

# For each NODEID, find the most common intersection name associated with it
nodeid_to_name = (
    crash_data.groupby("NODEID")["intersection_name"]
    .agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else None)
    .reset_index()
)
nodeid_to_name.columns = ["NODEID", "intersection_name"]

# Now match top 500 ranked names to NODEIDs
top_500_names = ranked_list["intersection_name"].head(500).tolist()

# Try both street orderings
nodeid_to_name["intersection_name_rev"] = (
    nodeid_to_name["intersection_name"].str.split(" & ").str[1].str.strip()
    + " & " +
    nodeid_to_name["intersection_name"].str.split(" & ").str[0].str.strip()
)

top_500_nodes_df = nodeid_to_name[
    nodeid_to_name["intersection_name"].isin(top_500_names) |
    nodeid_to_name["intersection_name_rev"].isin(top_500_names)
]

top_500_nodes = top_500_nodes_df["NODEID"].tolist()

# Filter crash_data
crash_data = crash_data[crash_data["NODEID"].isin(top_500_nodes)]

X = model_df[feature_cols].fillna(0)
y_text = model_df["primary_cause"]

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_text)

# X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
#     X, y, model_df.index,
#     test_size=0.2,
#     random_state=42,
#     stratify=y
# )

# Preprocessing: One-hot encode categorical features and standardize numeric features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), feature_cols)
    ]
)

# Pipeline: Preprocessor + Classifier
model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(
        n_estimators=500,
        max_depth=10,
        learning_rate=0.3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    ))
])

model.fit(X, y)
predictions = model.predict(X)
predicted_labels = label_encoder.inverse_transform(predictions)


# See results per intersection
results = model_df[["NODEID"]].copy()
results["predicted_cause"] = predicted_labels
results["actual_cause"] = y_text.values

results = results.merge(nodeid_to_name[["NODEID", "intersection_name"]], on="NODEID", how="left")
print(results[["intersection_name", "actual_cause", "predicted_cause"]])

# Evaluate model performance
score = f1_score(y, predictions, average="weighted", zero_division=0)
print(f"Weighted F1 Score: {score:.4f}") 
# 0.4590 when max_depth = 5
# 0.5963 when max_depth = 10
# 0.7289 when max_depth =10 and n_estimators = 500
# 0.7363 when max_depth =10, n_estimators = 500, and learning_rate = 0.3

mean_absolute_error_score = mean_absolute_error(y, predictions)
print(f"Mean Absolute Error: {mean_absolute_error_score:.4f}") 
# 1.2415 when max_depth = 5 since it is > 1, there is significant error in prediction
# 0.9205 when max_depth = 10 getting better
# 0.6280 when max_depth =10 and n_estimators = 500 
# 0.6065 when max_depth =10, n_estimators = 500, and learning_rate = 0.3

cause_labels = label_encoder.classes_.tolist()

confusion_matrix_result = confusion_matrix(y, predictions, labels=list(range(len(cause_labels))))

figure = px.imshow(
    confusion_matrix_result,
    labels=dict(x="Predicted", y="Actual"),
    x=cause_labels,
    y=cause_labels
)
figure.show()

# %%

