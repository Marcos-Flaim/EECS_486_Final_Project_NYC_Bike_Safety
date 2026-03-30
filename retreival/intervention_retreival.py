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

main()