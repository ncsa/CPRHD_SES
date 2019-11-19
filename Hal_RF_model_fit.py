
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
x = data.drop(columns=['wnvbinary', 'yrweeks', 'yrwksfid', 'yr_hexid', 'year'])

x_selected = x.drop(columns=x.columns[[4, 5, 25, 26, -17, -16, -15, -14, -13, -12, -11, -10, -9, -8, -6]]).values
y_selected = data['wnvbinary'].values
x = data.drop(columns=['wnvbinary', 'yrweeks', 'yrwksfid', 'yr_hexid', 'year'])
time_start = time.time()
trainX_sel, testX_sel, trainY_sel, testY_sel = train_test_split(x_selected, y_selected, test_size=0.2, random_state=1) # CV
print("data split:", time.time() - time_start)
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_selected), y=y_selected)
time_start = time.time()
model_RF_best_2 = RandomForestClassifier(n_estimators=2000,
                                         n_jobs=-1,
                                         max_features=5,
                                         max_depth=60,
                                         min_samples_leaf = 5,
                                         min_samples_split = 8,
                                         bootstrap=True,
                                        class_weight = 'balanced')
                                            
print("Classifier established:", time.time() - time_start)


time_start = time.time()
model_RF_best_2.fit(trainX_sel, trainY_sel)
print("model fit:", time.time() - time_start)

pickle.dump(model_RF_best_2, open('/home/jallen17/CPRHD_SES/RF_model_fit', 'wb'))
print("Fitting complete. Model saved as RF_model_fit")
