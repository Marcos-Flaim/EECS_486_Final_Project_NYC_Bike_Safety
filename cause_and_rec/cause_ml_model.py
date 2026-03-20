
# %%
import numpy
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error 
import pandas as pd 


# %%
raw_crash_data = pd.read_csv('crash_data.csv', low_memory=False)
node_map = pd.read_csv('crash_to_node_map.csv')
ranked_list = pd.read_csv('all_intersections_ranked.csv')

#add nodeid to raw crash data

raw_crash_data['COLLISION_ID'] = raw_crash_data['COLLISION_ID'].astype(int)
node_map[' COLLISION_ID'] = node_map['COLLISION_ID'].astype(int)

crash_data = raw_crash_data.merge(node_map, on='COLLISION_ID')

print(crash_data.columns)

# %%

# %%




