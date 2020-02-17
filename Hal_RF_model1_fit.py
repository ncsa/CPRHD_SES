import numpy as np
import time
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import log_loss
import pickle
from sklearn.utils import class_weight

"""
This code fits the Random Forest model. Cross validation is preformed in another
script.
"""

time_start = time.time()
data = pd.read_sas('/home/shared/cprhd/DATA_CPRHD_SES/wnv_2245new.sas7bdat')  # In the Cook_Dupage Directory
print("Data read in:", time.time() - time_start)

x = data.drop(columns=['yrweeks', 'yrwksfid', 'yr_hexid', 'year', 'income1','hexid','PopYesNo'])


columns = ['templag1', 'templag2', 'templag4', 'mirlag1', 'mirlag2',
       'mirlag3', 'mirlag4', 'totpop']
x_selected = x[columns].values
y_selected = x['wnvbinary'].values

trainX_sel, testX_sel, trainY_sel, testY_sel = train_test_split(x_selected, y_selected, test_size=0.2, random_state=1) # CV
print("data split:", time.time() - time_start)

model_RF_best_2 = RandomForestClassifier(n_estimators=1300,
                                         n_jobs=-1,
                                         max_depth=60,
                                         min_samples_leaf = 15,
                                         bootstrap=True)
                                            
print("Classifier established:", time.time() - time_start)



model_RF_best_2.fit(trainX_sel, trainY_sel)
print("model fit:", time.time() - time_start)

pickle.dump(model_RF_best_2, open('/home/jallen17/CPRHD_SES/RF_model_1_3_fit', 'wb'))
print("Fitting complete. Model saved as RF_model_fit")