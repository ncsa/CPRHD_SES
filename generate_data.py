import numpy as np
import pandas as pd

def double_check(dataframe):
    drop_columns = ["Id", "Geography_x", "Geography_y", "Id2_x", "Id2_y"]
    for drop_column in drop_columns:
        if drop_column in dataframe.columns:
            dataframe.drop(drop_column, 1, inplace=True)

# Read Features
features_dir = "./Features/"
features_2010 = pd.read_csv(features_dir + "features_2010_selected.csv", index_col = 0)
features_2011 = pd.read_csv(features_dir + "features_2011_selected.csv", index_col = 0)
features_2012 = pd.read_csv(features_dir + "features_2012_selected.csv", index_col = 0)
features_2013 = pd.read_csv(features_dir + "features_2013_selected.csv", index_col = 0)
features_2014 = pd.read_csv(features_dir + "features_2014_selected.csv", index_col = 0)
features_2015 = pd.read_csv(features_dir + "features_2015_selected.csv", index_col = 0)
features_2016 = pd.read_csv(features_dir + "features_2016_selected.csv", index_col = 0)
list_features = []
list_features.append(features_2010)
list_features.append(features_2011)
list_features.append(features_2012)
list_features.append(features_2013)
list_features.append(features_2014)
list_features.append(features_2015)
list_features.append(features_2016)

# Double check
for features in list_features:
    double_check(features)

# combination
# NOTICE : this may take LONG time

data = np.zeros(len(list_features[0].columns))

for i in range(len(list_features)):
    annual_tract_mir_index = i + 5

    annual_tract_mir = list_annual_tract_mir[annual_tract_mir_index]
    features_data = list_features[i]

    for item in annual_tract_mir.items():
        tract_id = int(item[0])
        mir_data = float(item[1])
        social_economic_data = features_data.loc[features_data["Id2"] == tract_id]
        # This tract may have been dropped in data cleaning
        if social_economic_data.empty:
            continue
        social_economic_data.drop("Id2", 1, inplace = True)

        row_data = np.concatenate((social_economic_data.to_numpy(), mir_data), axis = None)
        data = np.vstack([data, row_data])

# Write File
np.savetxt("data2.txt", data)
