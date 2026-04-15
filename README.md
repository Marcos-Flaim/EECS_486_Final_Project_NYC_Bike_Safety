# EECS_486_Final_Project_NYC_Bike_Safety:
This project identifies high-risk intersections in New York City using crash data, predicts the primary cause of crashes using machine learning, and recommends targeted safety interventions.

The pipeline combines:

* Data ingestion from NYC Open Data
* Geospatial intersection matching (LION dataset)
* Feature engineering + ML modeling (XGBoost)
* Semantic retrieval for intervention recommendations
* Interactive visualization via Streamlit

# Structure:

├── app.py                      # Streamlit dashboard
├── data/                       # All input/output CSVs (precomputed)
├── get_rankings/               # Crash data ingestion + intersection ranking
├── Boolean_List/               # Intersection feature engineering
├── cause_and_rec/              # ML model for crash cause prediction
├── retreival/                  # Intervention recommendation system


# Run Instructions:
1. Install dependencies:
    pip install -r requirements.txt

2. Run the dashboard:
    streamlit run app.py


**The above uses precomputed data in `/data`, scripts can be rerun as such:
# Re-running Scripts (optional)
1. Generate intersection rankings (~10min to generate):
    unzip get_rankings/nyclion.zip
    python -m get_rankings.intersections

* Pulls NYC crash data via API
* Matches to LION NodeIDs

- Output: data/intersection_rankings.csv


2. Generate boolean features (~several hrs):
    python Boolean_List/boolean_list.py

* Maps nearby properties to intersections

- Output: data/boolean_list_output.csv


3. Train ML model for crash causes (<1min):
    python -m cause_and_rec.cause_ml_model

* Uses cached crash data (`crash_data.parquet`)

- Output: data/intersection_predictions.csv


# Key Files

* `data/Master_Top_500_Intersections.csv` (base dataset)
* `data/intersection_predictions.csv` (ML predicted causes)
* `data/boolean_list_output.csv` (intersection features)
* `data/Intervention Document.csv` (intervention corpus)


File organization:
1) intersections.py
Dependencies: nyclion data (matches intersections on nodeid), NYC crash data
Process: Use requests.get to obtain NYC crash data, clean and turn into GeoDataFrame, join NODEID from LION data on street names, calculate severity score for each intersection and output ranked list of top 500 most “severe” intersections
2) Boolean_list.py
Dependencies: ranked intersection file
Process: use requests to obtain boolean data, use geolocation/haversine formula to determine which properties are associated with which intersection, append to ranked intersection file
3) cause_and_rec.py
4) predicted_fix.py 
Dependencies: pandas, numpy, sklearn.metrics.pairwise, sentence_transformers, csv
Inputs: intervention_document.csv (handmade file using data and sources to come up with interventions)
Process: make use of intervention_document.csv as the corpus of documents to be returned where each row is a document. Merge the boolean_list_output and intersection_prediction.csv at an intersection level per row. Then we produce a structured text detailing the most likely cause of a crash and the intersection features. Then for each row run cosine similarity against all possible intersections. 
