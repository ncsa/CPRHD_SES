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

time_start = time.time()
x = data.drop(columns=['yrweeks', 'yrwksfid', 'yr_hexid', 'year', 'income1','hexid','PopYesNo'])


x_small = x[(x.weeks >= 22) & (x.weeks <= 31)]
columns = ['templag3', 'templag4', 'precilag2', 'mirmean', 'mirlag1',
       'totpop', 'dlipct']
x_selected = x_small[columns].values
y_selected = x_small['wnvbinary'].values

time_start = time.time()
trainX_sel, testX_sel, trainY_sel, testY_sel = train_test_split(x_selected, y_selected, test_size=0.2, random_state=1) # CV
print("data split:", time.time() - time_start)
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_selected), y=y_selected)
time_start = time.time()
model_RF_best_2 = RandomForestClassifier(n_estimators=1300,
                                         n_jobs=-1,
                                         max_depth=60,
                                         min_samples_leaf = 15,
                                         bootstrap=True)
                                            
print("Classifier established:", time.time() - time_start)


time_start = time.time()
model_RF_best_2.fit(trainX_sel, trainY_sel)
print("model fit:", time.time() - time_start)

pickle.dump(model_RF_best_2, open('/home/jallen17/CPRHD_SES/RF_model_fit_2_3', 'wb'))
print("Fitting complete. Model saved as RF_model_fit")
