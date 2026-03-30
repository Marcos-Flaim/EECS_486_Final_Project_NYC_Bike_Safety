import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Load a pretrained Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")
# collect all the data from the csv
data = pd.read_csv("../data/Intervention Document.csv")
    
def readCSV():
    # extract the column containing the keywords and title
    keywords = data["Keywords"].tolist()
    titles = data["Title"].tolist()
    
    # concatenate title and keyword since title is too short and keyword lack narrative
    rows = []
    for i in range(len(keywords)):
        rows.append(titles[i] + ": " + keywords[i])
    
    return rows

def createEmbeddings():
    # The sentences to encode
    sentences = readCSV()

    # Calculate embeddings by calling model.encode()
    return model.encode(sentences)
def main():
    print("Initializing BERT Model and creating Document Embeddings...")
    doc_embeddings = createEmbeddings()
    
    # Load the predictions dataframe
    predictions_df = pd.read_csv("../data/intersection_predictions.csv")
    
    #  Get ONLY the unique predicted causes 
    unique_causes = predictions_df["predicted_cause"].unique()
    
    # Create a dictionary to store AI answers
    cause_to_intervention = {}
    
    print("\nCalculating best interventions for the unique crash causes...")
    for cause in unique_causes:
        # embed the input query
        query_embedding = model.encode([cause])
        
        # compute cosine similarity on stored embedded vectors
        similarities = cosine_similarity(query_embedding, doc_embeddings)
        
        # get the index of the highest score
        best_index = np.argmax(similarities[0])
        
        # look up that index in dataFrame to get the title
        best_title = data["Title"].iloc[best_index]
        
        # Save it to the dictionary
        cause_to_intervention[cause] = best_title
        print(f"Mapped '{cause}' -> '{best_title}'")
        
    # apply the dictionary to all 51,000 rows to create a new column
    predictions_df["Top_Intervention"] = predictions_df["predicted_cause"].map(cause_to_intervention)
    
    # Load Top 500 dataframe
    top_500_df = pd.read_csv("../data/intersection_rankings.csv")
    
    # merge the two datasets together using the shared "NODEID" column
    # use a 'left' merge to keep only the Top 500 intersections, but grab the new columns from predictions_df
    master_df = top_500_df.merge(predictions_df[["NODEID", "predicted_cause", "Top_Intervention"]], on="NODEID", how="left")
    
    # Save the final Master Dataframe
    master_df.to_csv("../data/Master_Top_500_Intersections.csv", index=False)
    print("\nSUCCESS! Saved 'Master_Top_500_Intersections.csv'.")
    print(master_df.head())


def evaluate_precision_at_k():
    print("\n--- RUNNING EVALUATION (P@K) ---")
    doc_embeddings = createEmbeddings()
    
    # Load predictions to get unique causes
    predictions_df = pd.read_csv("../data/intersection_predictions.csv")
    unique_causes = predictions_df["predicted_cause"].unique()
    
    # --- THE GROUND TRUTH MAPPING ---
    ground_truth = {
        "Failure to Yield Right-of-Way": [
            "Restrict Left Turns", 
            "Leading Pedestrian Interval (LPI)", 
            "No-Turn-on-Red Restrictions",
            "Protected Left-Turn Phase",
            "Install Marked Crosswalks"
        ],
        "Driver Inattention/Distraction": [
            "Turn Calming", 
            "Install Advance Warning Signals", 
            "Dynamic Speed Feedback Signs",
            "High-Visibility Crosswalk Markings",
            "Rectangular Rapid Flashing Beacons (RRFB)"
        ],
        "Backing Unsafely": [
            "Intersection Daylighting", 
            "Remove Visual Obstructions", 
            "Improve Street Lighting",
            "Corner Tightening / Reduced Turning Radius"
        ], 
        "Following Too Closely": [
            "Improve Signal Timing", 
            "High-Friction Surface Treatment", 
            "Install Advance Warning Signals",
            "Speed Humps or Speed Tables"
        ], 
        "Other Vehicular": [
            "Convert to Roundabout",
            "Add All-Way Stop Control",
            "Speed Humps or Speed Tables",
            "Reduce Speed Limits"
        ],
        "Other": [
            "Convert to Roundabout",
            "Add All-Way Stop Control",
            "Reduce Speed Limits"
        ]
    }
    
    k_values = [1, 3, 5]
    results = {cause: {f"P@{k}": 0.0 for k in k_values} for cause in unique_causes if cause in ground_truth}
    
    for cause in unique_causes:
        if cause not in ground_truth:
            continue
            
        query_embedding = model.encode([cause])
        similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
        
        # Sort indices by highest similarity score
        ranked_indices = np.argsort(similarities)[::-1]
        ranked_titles = data["Title"].iloc[ranked_indices].tolist()
        
        # Add this print statement to debug!
        print(f"\nFor '{cause}', BERT's actual Top 5 are: {ranked_titles[:5]}")
        # Get the valid interventions for this specific cause
        valid_interventions = ground_truth[cause]
        
        # Calculate P@K
        for k in k_values:
            top_k_titles = ranked_titles[:k]
            hits = sum(1 for title in top_k_titles if title in valid_interventions)
            results[cause][f"P@{k}"] = hits / k
            
    if not results:
        print("No matching causes found in ground_truth. Please update the dictionary.")
        return
        
    # Create DataFrame and calculate Macro-Average
    results_df = pd.DataFrame.from_dict(results, orient='index')
    results_df.index.name = 'Predicted Crash Cause'
    
    avg_row = results_df.mean().to_frame("Macro-Average").T
    final_df = pd.concat([results_df, avg_row])
    
    # Print the table formatted for Markdown/LaTeX
    print("\n--- PUBLICATION READY TABLE ---")
    print(final_df.to_markdown(floatfmt=".2f"))

if __name__ == "__main__":
    main()
    evaluate_precision_at_k()