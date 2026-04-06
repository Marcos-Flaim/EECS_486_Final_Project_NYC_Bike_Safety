import requests
import json

def main():
    
    Crash_Dataset= "" #json format of all rows in dataset
    i=0
    while(i<100): #allows potential for 4 million rows
        temp_str = requests.get("https://data.cityofnewyork.us/resource/h9gi-nx95.json?$limit=50000&$offset=" + str(i*50000)).text #get data from API
        if(temp_str != "[]\n"): #blank return logic
            Crash_Dataset = Crash_Dataset + temp_str[1:len(temp_str)-2] + "\n," #json string formatting
        i+=1
    Crash_Dataset = "[" + Crash_Dataset[:len(Crash_Dataset)-2] + "]" #json string formatting
    
    #create python dictionary for parsing
    crash_dictionary = json.loads(Crash_Dataset)
