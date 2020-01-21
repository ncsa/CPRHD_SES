import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score
import numpy as np
import time
import scipy.stats as stats
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LinearRegression
from scipy import stats
from sklearn.metrics import log_loss
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import matplotlib.pyplot as plt
from sklearn.metrics import (brier_score_loss, precision_score, recall_score,
                             f1_score)
from sklearn.utils import class_weight

"""
This code fits the Random Forest model. Cross validation is preformed in another
script.
"""

time_start = time.time()
data = pd.read_csv('/home/shared/cprhd/DATA_CPRHD_SES/Agg_mirdata.csv',index_col = False)  # Need to edit this line, the agg_mirdata in on the box, uploaded by shubham
print("Data read in:", time.time() - time_start)
agg = data.drop(columns= data.columns[[1,5,7,6,8,9,10,11,12,15,16,17,18,-1,-6]])
agg.iloc[3900]['wnvbinary'] = 1 # Only exceptions
agg = agg.drop(columns = 'hexid')
agg_wnv = agg[agg.wnvbinary == 1]
agg_0 = agg[agg.wnvbinary == 0]
l = []
for x in agg.columns:
    print(x + ': ' + str(stats.ks_2samp(agg_wnv[x], agg_0[x])[1]))
    value = stats.ks_2samp(agg_wnv[x], agg_0[x])[1]
    if(value < 0.01):
        l.append(x)
data = agg[l]
x_selected = data.drop(columns = 'wnvbinary')
y_selected = data['wnvbinary'].values

trainX_sel, testX_sel, trainY_sel, testY_sel = train_test_split(x_selected, y_selected, test_size=0.2, random_state=1) # CV
print("data split:", time.time() - time_start)
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_selected), y=y_selected)
time_start = time.time()
model_RF_best_2 = RandomForestClassifier(n_estimators=1400,
                                         n_jobs=-1,
                                         max_features='sqrt',
                                         max_depth=80,
                                         min_samples_leaf = 8,
                                         min_samples_split = 6,
                                         bootstrap=True)
                                            
print("Classifier established:", time.time() - time_start)


time_start = time.time()
model_RF_best_2.fit(trainX_sel, trainY_sel)
print("model fit:", time.time() - time_start)

pickle.dump(model_RF_best_2, open('/home/jallen17/CPRHD_SES/RF_agg_model_fit', 'wb'))
print("Fitting complete. Model saved as RF_model_fit")

