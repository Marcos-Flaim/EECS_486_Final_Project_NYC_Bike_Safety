
# %%
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd 
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, classification_report, accuracy_score, confusion_matrix, f1_score
from xgboost import XGBClassifier
import plotly.express as px
from get_rankings.crash_data_loader import load_crash_data


# %%
#DOWNLOAD DATA AND MERGE NODE ID WITH THE CRASH DATA 

raw_crash_data = load_crash_data()
node_map = pd.read_csv("../data/crash_to_node_map.csv")
ranked_list = pd.read_csv("../data/intersection_rankings.csv")

#clean headers
raw_crash_data.columns = raw_crash_data.columns.str.strip()
node_map.columns = node_map.columns.str.strip()
ranked_list.columns = ranked_list.columns.str.strip().str.lower()

# Get top 500 ranked intersections
top_intersections = ranked_list.head(500)["nodeid"].tolist()

#add nodeid to raw crash data

raw_crash_data["collision_id"] = pd.to_numeric(raw_crash_data["collision_id"], errors="coerce")
node_map["collision_id"] = pd.to_numeric(node_map["collision_id"], errors="coerce")

raw_crash_data = raw_crash_data.dropna(subset=["collision_id"])
node_map = node_map.dropna(subset=["collision_id"])

raw_crash_data["collision_id"] = raw_crash_data["collision_id"].astype("int64")
node_map["collision_id"] = node_map["collision_id"].astype("int64")

#merge crash rows to NODEID
crash_data = raw_crash_data.merge(node_map[["collision_id", "NODEID"]], on="collision_id", how="inner")
#remove duplicate columns
crash_data = crash_data.loc[:, ~crash_data.columns.duplicated()]
print("Merged rows:", len(crash_data))
print("Unique intersections:", crash_data["NODEID"].nunique())



# %%
#CREATE FEATURES 

#convert crash date to a datetime, 
# and make some booleans for pedestrian injury etc. 
#boolean about rush hourmonth, weekend, and night

crash_data["crash_date"] = pd.to_datetime(crash_data["crash_date"], errors="coerce")
crash_data["crash_time"] = pd.to_datetime(crash_data["crash_time"], format="%H:%M", errors="coerce")

crash_data["hour"] = crash_data["crash_time"].dt.hour
crash_data["month"] = crash_data["crash_date"].dt.month
crash_data["year"] = crash_data["crash_date"].dt.year
crash_data["dayofweek"] = crash_data["crash_date"].dt.dayofweek

crash_data["is_night"] = ((crash_data["hour"] >= 20) | (crash_data["hour"] <= 5)).astype(int)
crash_data["is_rush_hour"] = crash_data["hour"].isin([7, 8, 9, 16, 17, 18]).astype(int)
crash_data["is_weekend"] = (crash_data["dayofweek"] >= 5).astype(int)

crash_data["is_pedestrian_injury"] = (crash_data["number_of_pedestrians_injured"] > 0).astype(int)
crash_data["is_pedestrian_death"] = (crash_data["number_of_pedestrians_killed"] > 0).astype(int)
crash_data["is_bike_injury"] = (crash_data["number_of_cyclist_injured"] > 0).astype(int)
crash_data["is_bike_death"] = (crash_data["number_of_cyclist_killed"] > 0).astype(int)
crash_data["is_motorist_injury"] = (crash_data["number_of_motorist_injured"] > 0).astype(int)
crash_data["is_fatal"] = (crash_data["number_of_persons_killed"] > 0).astype(int)

crash_data["is_severe"] = (
    (crash_data["number_of_persons_killed"] > 0) |
    (crash_data["number_of_persons_injured"] >= 2)
).astype(int)

# %%
#USE CONTRIBUTING FACTOR VEHICLE 1 BECUASE IT IS LESS NOISEY .
#WE CAN ADD CONTR. FAC. 2-4 LATER TO C IF IT IMPROVES MODEL 

cause_col = "contributing_factor_vehicle_1"
crash_data[cause_col] = crash_data[cause_col].fillna("Unspecified").astype(str).str.strip()
# Before creating cause_label, filter out rows where cause was "Unspecified"
crash_data = crash_data[crash_data[cause_col] != "Unspecified"]

#only keep top causes LESS NOISY 
top_causes = crash_data[cause_col].value_counts().nlargest(5).index.tolist()
print("Top causes:", top_causes)

# delete them from the dataframe completely during your preprocessing step
useless_labels = ["Unspecified", "Other Vehicular", "Following Too Closely"]
crash_data = crash_data[~crash_data[cause_col].isin(useless_labels)]

crash_data["cause_label"] = crash_data[cause_col].apply(
    lambda x: x if x in top_causes else "Other"
)

print("\nCause label counts:")
print(crash_data["cause_label"].value_counts())

# %%
#AGGREGATE FEATURES BY THE INTERSECTION OR NODEIDE
# AND CREATE NUMERIC FEATURES /RATIOS BC IT WORLS BETTER WITH XGB 
model_df = crash_data.groupby("NODEID").agg(
    total_crashes=("collision_id", "count"),
    total_injured=("number_of_persons_injured", "sum"),
    total_killed=("number_of_persons_killed", "sum"),
    ped_injury_count=("is_pedestrian_injury", "sum"),
    ped_death_count=("is_pedestrian_death", "sum"),
    bike_injury_count=("is_bike_injury", "sum"),
    bike_death_count=("is_bike_death", "sum"),
    night_crash_count=("is_night", "sum"),
    rush_crash_count=("is_rush_hour", "sum"),
    weekend_crash_count=("is_weekend", "sum"),
    fatal_crash_count=("is_fatal", "sum"),
    avg_injured=("number_of_persons_injured", "mean")
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
        n_estimators=300,        # Reduced to prevent memorization
        max_depth=6,             # Reduced to force general rule-learning
        learning_rate=0.1,       # Slowed down so it learns minority classes carefully
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,      # NEW: Prevents hyper-specific edge cases
        gamma=0.1,               # NEW: Penalizes the model for making useless tree branches
        random_state=42
    ))
])

# Sam: Calculate weights: Rare classes get high weights, common classes get low weights
# Get the standard balanced weights for each unique class
classes = np.unique(y)
raw_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)

# "Soften" the weights using a square root to prevent over-correction
soft_weights = np.sqrt(raw_weights)
weight_dict = {c: w for c, w in zip(classes, soft_weights)}

# Map these soft weights to every single row in your dataset
sample_weights = np.array([weight_dict[label] for label in y])
# Pass the weights to the XGBoost step inside your pipeline
# The 'classifier__' prefix tells the pipeline to send this specifically to XGBoost
model.fit(X, y, classifier__sample_weight=sample_weights)

predictions = model.predict(X)
predicted_labels = label_encoder.inverse_transform(predictions)


# See results per intersection
results = model_df[["NODEID"]].copy()
results["predicted_cause"] = predicted_labels
results["actual_cause"] = y_text.values

# Now that predictions are made:
# Filter model_df to only include top 500 intersections
model_df = model_df[model_df["NODEID"].isin(top_intersections)]
results = results[results["NODEID"].isin(top_intersections)]

results = results.merge(
    ranked_list[["nodeid", "intersection_name"]],
    left_on="NODEID",
    right_on="nodeid",
    how="left"
)
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
# sam: commented this out because I found the pop up annoying
# figure.show()

# sam: need to have the data somewhere
results.to_csv("../data/intersection_predictions.csv", index=False)
print("Saved predictions to ../data/intersection_predictions.csv")

# %%

