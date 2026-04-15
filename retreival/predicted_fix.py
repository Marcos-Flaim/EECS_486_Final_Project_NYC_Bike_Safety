import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import csv

# Load a pretrained Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

Intervention_Document = pd.read_csv("data/Intervention Document.csv")

def readCSV():
    # extract the column containing the keywords and title
    keywords = Intervention_Document["Keywords"].tolist()
    titles = Intervention_Document["Title"].tolist()
    
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
    boolean_list = open("data/boolean_list_output.csv", "rt")
    intersection_predictions = open("data/intersection_predictions.csv", "rt")
    
    output_file = open("data/predicted_fix.csv", "w", newline= '')
    output_file.write("NODEID,crash_count,total_killed,total_injured,severity_score,severity_score_norm,intersection_name,latitude,longitude,Has_Enhanced_Crossing,Has_Leading_Pedestrian_Signal,Has_Turn_Traffic_Calming,Has_SIP,Has_Exclusive_Pedestrian_Signal,Has_Accessible_Pedestrian_Signal,Is_Protected_Intersection,Has_Pedestrian_Ramp,Has_Bus_Lane,Has_Speed_Hump,Is_Speed_Reduced,Is_Bike_Route,Has_Bad_Pavement,Predicted_Fix\n")
    output_list = []
    
    csv_boolean_list_parse = csv.reader(boolean_list)
    csv_predictions_list_parse = csv.reader(intersection_predictions)
    
    # data from boolean list, used for embeddings
    cols = ["Has enhanced crossing", "Has leading pedestrian signal", "Has turn traffic calming", "has street improvment project", "has exclusive pedestrian signal", "has accessible pedestrian signal", "is protected intersection", "Has pedestrian ramp", "has bus lane", "has speed hump", "has speed reduction", "is bike route", "has bad pavement"]
    ops_cols = ["Does not have enhanced crossing", "Does not have leading pedestrian signal", "Does not have turn traffic calming", "does not have street improvment project", "Does not have exclusive pedestrian signal", "Does not have accessible pedestrian signal", "Does not have protected intersection", "Does not have pedestrian ramp", "Does not have bus lane", "Does not have speed hump", "Does not have speed reduction", "Does not have bike route", "Does not have bad pavement"]
    
    intervention_embeddings = createEmbeddings()
    
    i=0
    for ranked_intersection in csv_boolean_list_parse:
        if(i==0): 
            i+=1 
            continue
        
        # create intersection embedding string 
        embedding_string = ""
        start_index = 9
        while(start_index<22):
            #if(ranked_intersection[start_index]== "0"): embedding_string += ops_cols[start_index-9] + ", "
            #else: embedding_string += cols[start_index-9] + ", "

            if(ranked_intersection[start_index]== "1"): embedding_string += cols[start_index-9] + ", "
            start_index += 1
        j=0
        for node in csv_predictions_list_parse:
            if(j==0): 
                j+=1
                continue
            # find predicted cause
            if(int(node[0])==int(ranked_intersection[0])): #node ID's are equal
                embedding_string += ", Predicted cause is " + node[1]
            j+=1
        
        query_embedding = model.encode([embedding_string])
        
         # compute cosine similarity on stored embedded vectors
        similarities = cosine_similarity(query_embedding, intervention_embeddings)
        
        # get the index of the highest score
        best_index = np.argmax(similarities[0])
        
        # look up that index in dataFrame to get the title
        best_title = Intervention_Document.iloc[best_index]["Title"]
        ranked_intersection.append(best_title)
        
        #output logic
        output_list.append(ranked_intersection)
        
        i+=1
    
    #csv output
    writer = csv.writer(output_file)
    writer.writerows(output_list)
    
main()
