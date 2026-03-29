import csv
import requests
import json
from haversine import haversine

def within_15_meters(lat1, long1, lat2, long2):
    return (haversine((lat1,long1),(lat2,long2))*1000)<15
    

def main():
    #create list(of lists) with intersections and their properties(booleans)
    ranked_intersections_file = open("Top_500_Intersections.csv", "rt")
    csv_parse = csv.reader(ranked_intersections_file)
    output = []
    j=0
    for row in csv_parse:
        if(j!=0):
            i=0
            while i<4:
                row.append("0")
                i+=1
            output.append(row)
        j+=1
        
    VZV_Enhanced_Crossings_JSON = requests.get("https://data.cityofnewyork.us/resource/6ax4-q5k4.json").text
    VZV_Leading_Pedestrian_Interval_Signals_JSON = requests.get("https://data.cityofnewyork.us/resource/xc4v-ntf4.json").text
    VZV_Turn_Traffic_Calming_JSON = requests.get("https://data.cityofnewyork.us/resource/sm2x-35i7.json").text
    VZV_Street_Improvement_Projects_JSON = requests.get("https://data.cityofnewyork.us/resource/shr7-eqdc.json").text
    
    VZV_Enhanced_Crossings_Dictionary = json.loads(VZV_Enhanced_Crossings_JSON)
    VZV_Leading_Pedestrian_Interval_Signals_Dictionary = json.loads(VZV_Leading_Pedestrian_Interval_Signals_JSON)
    VZV_Turn_Traffic_Calming_Dictionary = json.loads(VZV_Turn_Traffic_Calming_JSON)
    VZV_Street_Improvement_Projects_Dictionary = json.loads(VZV_Street_Improvement_Projects_JSON)
    
    for x in output:
        for y in VZV_Enhanced_Crossings_Dictionary:
            if((y.get("lat")!=None) and (y.get("long")!=None)):
                if(within_15_meters(float(x[7]),float(x[8]), float(y["lat"]),float(y["long"]))):
                    x[9] = "1"
                    break
        for y in VZV_Leading_Pedestrian_Interval_Signals_Dictionary:
            if((y.get("lat")!=None) and (y.get("long")!=None)):
                if(within_15_meters(float(x[7]),float(x[8]), float(y["lat"]),float(y["long"]))):
                    x[10] = "1"
                    break
        for y in VZV_Turn_Traffic_Calming_Dictionary:
            if((y.get("lat")!=None) and (y.get("long")!=None)):
                if(within_15_meters(float(x[7]),float(x[8]), float(y["lat"]),float(y["long"]))):
                    x[11] = "1"
                    break
        for y in VZV_Street_Improvement_Projects_Dictionary:
            if((y.get("lat")!=None) and (y.get("long")!=None)):
                if(within_15_meters(float(x[7]),float(x[8]), float(y["lat"]),float(y["long"]))):
                    x[12] = "1"
                    break
    
    output_file = open("boolean_list_output.csv", "w")
    output_file.write("NODEID,crash_count,total_killed,total_injured,severity_score,severity_score_norm,intersection_name,latitude,longitude,Has_Enhanced_Crossing,Has_Leading_Pedestrian_Signal,Has_Turn_Traffic_Calming,Has_SIP\n")
    writer = csv.writer(output_file)
    writer.writerows(output)
    
main()
