Use requests.get to obtain NYC crash data, clean and turn into
GeoDataFrame, join NODEID from LION data on street names,
calculate severity score for each intersection and output
ranked list of top 500 most “severe” intersections

(geotest.py and importtest.py were earlier iterations.
can be found in 'iterations' folder)

Edit: Ensure these downloads are in /data folder

- NYC LION data (name folder lion)
https://data.cityofnewyork.us/City-Government/LION/2v4z-66xt/about_data

To run:
(1) Ensure you're in .../EECS_486_Final_Project_NYC_Bike_Safety/get_rankings
(2) Run mamba activate streets in terminal
(3) Install any necessary libraries
(4) Run: unzip nyclion.zip (to unzip nyclion files)
(5) Run: python intersections.py (outputs intersection rankings to csv)

