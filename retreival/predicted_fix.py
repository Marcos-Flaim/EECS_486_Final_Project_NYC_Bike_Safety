import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import csv

# Load a pretrained Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

def main():
    boolean_list = open("../data/boolean_list_output.csv", "rt")
    intersection_predictions = open("../data/intersection_predictions.csv", "rt")
    Intervention_Document = open("../data/Intervention Document.csv", "rt")
    
    csv_boolean_list_parse = csv.reader(boolean_list)
    csv_predictions_list_parse = csv.reader(intersection_predictions)
    
    # data from boolean list, used for embeddings
    cols = ["Has enhanced crossing", "Has leading pedestrian signal", "Has turn traffic calming", "has exclusive pedestrian signal", "has accessible pedestrian signal", "is protected intersection", "Has pedestrian ramp", "has bus lane", "has speed hump", "has speed reduction", "is bike route", "has bad pavement"]
    ops_cols = ["Does not have enhanced crossing", "Does not have leading pedestrian signal", "Does not have turn traffic calming", "Does not have exclusive pedestrian signal", "Does not have accessible pedestrian signal", "Does not have protected intersection", "Does not have pedestrian ramp", "Does not have bus lane", "Does not have speed hump", "Does not have speed reduction", "Does not have bike route", "Does not have bad pavement"]
    
    i=0
    for ranked_intersection in csv_boolean_list_parse:
        if(i==0): continue
        # create intersection embedding string 
        embedding_string = ""
        start_index = 9
        while(start_index<22):
            if(ranked_intersection[start_index]== "0"): embedding_string += cols[start_index-9] + ", "
            else: embedding_string += ops_cols[start_index-9] + ", "
            start_index += 1
        j=0
        for node in csv_predictions_list_parse:
            if(j==0): continue
            # find predicted cause
            if(int(node[0])==int(ranked_intersection[0])): #node ID's are equal
                
            j+=1
        
        i+=1
    
main()