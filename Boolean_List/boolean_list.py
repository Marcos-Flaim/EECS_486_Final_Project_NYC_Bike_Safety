import csv
import requests
import json
import math
from haversine import haversine

def within_15_meters(lat1, long1, lat2, long2):
    lat1 = math.fabs(lat1)
    long1 = math.fabs(long1)
    lat2 = math.fabs(lat2)
    long2 = math.fabs(long2)
    return (haversine((lat1,long1),(lat2,long2))*1000)<50

def within_15_meters_line(lat1, long1, pair1, pair2):
    lat1 = math.fabs(lat1)
    long1 = math.fabs(long1)
    pair1[0] = math.fabs(pair1[0])
    pair1[1] = math.fabs(pair1[1])
    pair2[0] = math.fabs(pair2[0])
    pair2[1] = math.fabs(pair2[1])
    line_distance = int(haversine((pair1[0],pair1[1]),(pair2[0],pair2[1]))*1000)
    i=0
    while(i<line_distance):
        new_lat = math.fabs(((math.fabs(pair1[1]-pair2[1])/line_distance)*i) + min(pair1[1],pair2[1]))
        new_long = math.fabs(((math.fabs(pair1[0]-pair2[0])/line_distance)*i) + min(pair1[0],pair2[0]))
        if(within_15_meters(lat1, long1, new_lat, new_long)):
            return True
        i+=10
    return False
        
    

def main():
    #create list(of lists) with intersections and their properties(booleans)
    ranked_intersections_file = open("data/Top_500_Intersections.csv", "rt")
    csv_parse = csv.reader(ranked_intersections_file)
    output = []
    j=0
    for row in csv_parse:
        if(j!=0):
            i=0
            while i<13:
                row.append("0")
                i+=1
            output.append(row)
        j+=1
    
    
    #point <50000
    Exclusive_Pedestrian_Signal_Locations_JSON = requests.get("https://data.cityofnewyork.us/resource/8kuj-2n3u.json?$limit=50000").text
    Accessible_Pedestrian_Signal_Locations_JSON = requests.get("https://data.cityofnewyork.us/resource/de3m-c5p4.json?$limit=50000").text
    VZV_Enhanced_Crossings_JSON = requests.get("https://data.cityofnewyork.us/resource/6ax4-q5k4.json?$limit=50000").text
    VZV_Leading_Pedestrian_Interval_Signals_JSON = requests.get("https://data.cityofnewyork.us/resource/xc4v-ntf4.json?$limit=50000").text
    VZV_Turn_Traffic_Calming_JSON = requests.get("https://data.cityofnewyork.us/resource/sm2x-35i7.json?$limit=50000").text
    VZV_Street_Improvement_Projects_JSON = requests.get("https://data.cityofnewyork.us/resource/shr7-eqdc.json?$limit=50000").text
    
    #point >50000
    Pedestrian_Ramp_Locations_JSON = ""
    i=0
    while(i<100):
        temp_str = requests.get("https://data.cityofnewyork.us/resource/ufzp-rrqu.json?$limit=50000&$offset="+str(i*50000)).text
        if(temp_str != "[]\n"):
            Pedestrian_Ramp_Locations_JSON = Pedestrian_Ramp_Locations_JSON + temp_str[1:len(temp_str)-2] + "\n,"
        i+=1
    Pedestrian_Ramp_Locations_JSON = "[" + Pedestrian_Ramp_Locations_JSON[:len(Pedestrian_Ramp_Locations_JSON)-2] + "]"
    
    #multiline >50000
    Speed_Reducer_Tracking_System_JSON = ""
    i=0
    while(i<100):
        temp_str = requests.get("https://data.cityofnewyork.us/resource/9n6h-pt9g.json?$limit=50000&$offset="+str(i*50000)).text
        if(temp_str != "[]\n"):
            Speed_Reducer_Tracking_System_JSON = Speed_Reducer_Tracking_System_JSON + temp_str[1:len(temp_str)-2] + "\n,"
        i+=1
    Speed_Reducer_Tracking_System_JSON = "[" + Speed_Reducer_Tracking_System_JSON[:len(Speed_Reducer_Tracking_System_JSON)-2] + "]"
    
    Pavement_Ratings_JSON = ""
    i=0
    while(i<100):
        temp_str = requests.get("https://data.cityofnewyork.us/resource/6yyb-pb25.json?$limit=50000&$offset="+str(i*50000)).text
        if(temp_str != "[]\n"):
            Pavement_Ratings_JSON = Pavement_Ratings_JSON + temp_str[1:len(temp_str)-2] + "\n,"
        i+=1
    Pavement_Ratings_JSON = "[" + Pavement_Ratings_JSON[:len(Pavement_Ratings_JSON)-2] + "]"
    
    #multipoint <50000
    Protected_Streets_Intersection_JSON = requests.get("https://data.cityofnewyork.us/resource/bryy-vqd9.json?$limit=50000").text
    
    #multiline <50000
    Bus_lanes_JSON = requests.get("https://data.cityofnewyork.us/resource/ycrg-ses3.json?$limit=50000").text
    Speed_Humps_JSON = requests.get("https://data.cityofnewyork.us/resource/jknp-skuy.json?$limit=50000").text
    Bike_Routes_JSON = requests.get("https://data.cityofnewyork.us/resource/mzxg-pwib.json?$limit=50000").text
    
    
    #processing json into usable dictionaries
    VZV_Enhanced_Crossings_Dictionary = json.loads(VZV_Enhanced_Crossings_JSON)
    VZV_Leading_Pedestrian_Interval_Signals_Dictionary = json.loads(VZV_Leading_Pedestrian_Interval_Signals_JSON)
    VZV_Turn_Traffic_Calming_Dictionary = json.loads(VZV_Turn_Traffic_Calming_JSON)
    VZV_Street_Improvement_Projects_Dictionary = json.loads(VZV_Street_Improvement_Projects_JSON)
    Exclusive_Pedestrian_Signal_Locations_Dictionary = json.loads(Exclusive_Pedestrian_Signal_Locations_JSON)
    Accessible_Pedestrian_Signal_Locations_Dictionary = json.loads(Accessible_Pedestrian_Signal_Locations_JSON)
    Protected_Streets_Intersection_Dictionary = json.loads(Protected_Streets_Intersection_JSON)
    Pedestrian_Ramp_Locations_Dictionary = json.loads(Pedestrian_Ramp_Locations_JSON)
    Speed_Reducer_Tracking_System_Dictionary = json.loads(Speed_Reducer_Tracking_System_JSON)
    Pavement_Ratings_Dictionary = json.loads(Pavement_Ratings_JSON)
    Bus_lanes_Dictionary = json.loads(Bus_lanes_JSON)
    Speed_Humps_Dictionary = json.loads(Speed_Humps_JSON)
    Bike_Routes_Dictionary = json.loads(Bike_Routes_JSON) 
    
    j=1
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
        for y in Exclusive_Pedestrian_Signal_Locations_Dictionary:
            if((y.get("the_geom")!=None) and (y["the_geom"].get("coordinates")!=None) and (len(y["the_geom"]["coordinates"])==2) and (str(type(y["the_geom"]["coordinates"][0]))=="<class 'float'>")):
                if(within_15_meters(float(x[7]),float(x[8]),y["the_geom"]["coordinates"][1],y["the_geom"]["coordinates"][0])):
                    x[13] = "1"
                    break
        for y in Accessible_Pedestrian_Signal_Locations_Dictionary:
            if((y.get("the_geom")!=None) and (y["the_geom"].get("coordinates")!=None) and (len(y["the_geom"]["coordinates"])==2) and (str(type(y["the_geom"]["coordinates"][0]))=="<class 'float'>")):
                if(within_15_meters(float(x[7]),float(x[8]),y["the_geom"]["coordinates"][1],y["the_geom"]["coordinates"][0])):
                    x[14] = "1"
                    break
        for y in Protected_Streets_Intersection_Dictionary:
            if((y.get("the_geom")!=None) and (y["the_geom"].get("coordinates")!=None) and (len(y["the_geom"]["coordinates"])==2) and (str(type(y["the_geom"]["coordinates"][0]))=="<class 'float'>")):
                if(within_15_meters(float(x[7]),float(x[8]),y["the_geom"]["coordinates"][1],y["the_geom"]["coordinates"][0])):
                    x[15] = "1"
                    break
            elif((y.get("the_geom")!=None) and (y["the_geom"].get("coordinates")!=None)):
                for z in y["the_geom"]["coordinates"]:
                    if(within_15_meters(float(x[7]),float(x[8]),z[1],z[0])):
                        x[15] = "1"
                        break 
        for y in Pedestrian_Ramp_Locations_Dictionary:
            if((y.get("the_geom")!=None) and (y["the_geom"].get("coordinates")!=None) and (len(y["the_geom"]["coordinates"])==2) and (str(type(y["the_geom"]["coordinates"][0]))=="<class 'float'>")):
                if(within_15_meters(float(x[7]),float(x[8]),y["the_geom"]["coordinates"][1],y["the_geom"]["coordinates"][0])):
                    x[16] = "1"
                    break
        for y in Bus_lanes_Dictionary:
            if((y.get("the_geom")!=None) and (y["the_geom"].get("coordinates")!=None)):
                i=0
                while(i<(len(y["the_geom"]["coordinates"][0])-1)):
                    if(within_15_meters_line(math.fabs(float(x[7])),math.fabs(float(x[8])),y["the_geom"]["coordinates"][0][i+1],y["the_geom"]["coordinates"][0][i])):
                        x[17] = "1"
                        break
                    i+=1
        for y in Speed_Humps_Dictionary:
            if((y.get("the_geom")!=None) and (y["the_geom"].get("coordinates")!=None)):
                i=0
                while(i<(len(y["the_geom"]["coordinates"][0])-1)):
                    if(within_15_meters_line(math.fabs(float(x[7])),math.fabs(float(x[8])),y["the_geom"]["coordinates"][0][i+1],y["the_geom"]["coordinates"][0][i])):
                        x[18] = "1"
                        break
                    i+=1
        for y in Speed_Reducer_Tracking_System_Dictionary:
            if((y.get("fromlatitude")!= None) and (y.get("fromlongitude")!= None) and (y.get("tolatitude")!= None) and (y.get("tolongitude")!= None)):
                if(within_15_meters_line(math.fabs(float(x[7])),math.fabs(float(x[8])),[math.fabs(float(y["fromlatitude"])),math.fabs(float(y["fromlongitude"]))],[math.fabs(float(y["tolatitude"])),math.fabs(float(y["tolongitude"]))])):
                    x[19] = "1"
                    break
        for y in Bike_Routes_Dictionary:
            if((y.get("the_geom")!=None) and (y["the_geom"].get("coordinates")!=None)):
                i=0
                while(i<(len(y["the_geom"]["coordinates"][0])-1)):
                    if(within_15_meters_line(math.fabs(float(x[7])),math.fabs(float(x[8])),y["the_geom"]["coordinates"][0][i+1],y["the_geom"]["coordinates"][0][i])):
                        x[20] = "1"
                        break
                    i+=1
        for y in Pavement_Ratings_Dictionary:
            if((y.get("the_geom")!=None) and (y["the_geom"].get("coordinates")!=None) and (y.get("systemrating")!=None)):
                i=0
                while(i<(len(y["the_geom"]["coordinates"][0])-1)):
                    if(within_15_meters_line(math.fabs(float(x[7])),math.fabs(float(x[8])),y["the_geom"]["coordinates"][0][i+1],y["the_geom"]["coordinates"][0][i])):
                        if(float(y["systemrating"])<3.5):
                            x[21] = "1"
                            break
                    i+=2
        print("Intersection " + str(j) + " parsed\n")
        j+=1
    
    output_file = open("data/boolean_list_output.csv", "w")
    output_file.write("NODEID,crash_count,total_killed,total_injured,severity_score,severity_score_norm,intersection_name,latitude,longitude,Has_Enhanced_Crossing,Has_Leading_Pedestrian_Signal,Has_Turn_Traffic_Calming,Has_SIP,Has_Exclusive_Pedestrian_Signal,Has_Accessible_Pedestrian_Signal,Is_Protected_Intersection,Has_Pedestrian_Ramp,Has_Bus_Lane,Has_Speed_Hump,Is_Speed_Reduced,Is_Bike_Route,Has_Bad_Pavement\n")
    writer = csv.writer(output_file)
    writer.writerows(output)
    
main()
