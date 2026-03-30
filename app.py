import streamlit as st
import pandas as pd
import plotly.express as px

# Configure the page settings
st.set_page_config(page_title="NYC Street Safety", layout="wide", page_icon="🚲")

# Add a Title and Description
st.title("🚲 NYC Street Safety & Intervention Dashboard")
st.markdown("""
This dashboard identifies high-risk intersections in New York City. 
It utilizes **XGBoost** to predict the primary cause of crashes and **Sentence-BERT** to recommend targeted street safety interventions.
""")

# Load the Master Dataframe
@st.cache_data
def load_data():
    df = pd.read_csv("data/Master_Top_500_Intersections.csv")
    
    # Sort by severity score by default
    df = df.sort_values(by="severity_score", ascending=False)
    return df

df = load_data()

# Create a Sidebar for interactive filtering
st.sidebar.header("Filter Options")

# Let users filter the map by the AI's predicted crash cause
unique_causes = df["predicted_cause"].unique()
selected_cause = st.sidebar.selectbox("Filter by Predicted Cause", ["All"] + list(unique_causes))

if selected_cause != "All":
    filtered_df = df[df["predicted_cause"] == selected_cause]
else:
    filtered_df = df

# Build the Interactive Map (Heatmap)
st.subheader("High-Risk Intersections Map")

# use Plotly Express to plot the latitude/longitude from your Top 500 CSV
fig = px.scatter_mapbox(
    filtered_df, 
    lat="latitude", 
    lon="longitude",
    color="severity_score",         # Color the dots based on severity
    size="crash_count",             # Make dots bigger if there are more crashes
    hover_name="intersection_name", # Bold title when hovering
    hover_data={                    # What to show in the tooltip
        "latitude": False, 
        "longitude": False,
        "predicted_cause": True, 
        "Top_Intervention": True, 
        "severity_score": True
    },
    color_continuous_scale=px.colors.sequential.YlOrRd, # Yellow to Orange to Red
    zoom=10, 
    mapbox_style="carto-positron"   # A clean, modern map style
)

# Render the map in Streamlit
st.plotly_chart(fig, use_container_width=True)

# Build the Ranked Leaderboard (Data Table)
st.subheader("Prioritized Intervention Rankings")
st.markdown("Click on any column header to sort the data.")

# Select only the most important columns to show to the user
display_cols = [
    "intersection_name", 
    "severity_score", 
    "crash_count", 
    "predicted_cause", 
    "Top_Intervention"
]

# Display as an interactive dataframe
st.dataframe(filtered_df[display_cols], use_container_width=True, hide_index=True)