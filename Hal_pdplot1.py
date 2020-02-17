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
from sklearn.inspection import plot_partial_dependence
import pickle

model = pickle.load(open('/home/shared/cprhd/MODELS/RF_model_1_2_fit','rb')) # Change this line to new model -- RF_model_CV_final_11-27
data = pd.read_sas('/home/shared/cprhd/DATA_CPRHD_SES/wnv_2245new.sas7bdat')
x = data.drop(columns=['yrweeks', 'yrwksfid', 'yr_hexid', 'year', 'income1','hexid','PopYesNo'])

columns = ['tempc', 'preci', 'templag1', 'templag2',
       'templag3', 'templag4', 'precilag1', 'precilag2', 'precilag3',
       'precilag4', 'mirmean', 'mirlag1', 'mirlag2', 'mirlag3', 'mirlag4',
       'totpop', 'dlipct', 'dmipct', 'Jantemp', 'hpctpreww', 'hpctpostww',
       'hpct7089']
x_selected = x[columns].values
y_selected = x['wnvbinary'].values

trainX_sel, testX_sel, trainY_sel, testY_sel = train_test_split(x_selected, y_selected, test_size=0.2, random_state=1) # CV

fig_1 = plt.figure()
time_start = time.time()
features = [3,7]
column_names = columns

plot_partial_dependence(model, trainX, features, feature_names=column_names,fig= fig_1)
fig_1.savefig('result1.png')
print("Time consumed:", time.time() - time_start)

fig_2 = plt.figure()
time_start = time.time()
features = [12,15]
plot_partial_dependence(model, trainX, features, feature_names=column_names,fig= fig_2)
fig_2.savefig('result2.png')
print("Time consumed:", time.time() - time_start)

fig_3 = plt.figure()
time_start = time.time()
features = [16,20]
plot_partial_dependence(model, trainX, features, feature_names=column_names,fig= fig_3)
fig_3.savefig('result3.png')
print("Time consumed:", time.time() - time_start)

fig_4 = plt.figure()
time_start = time.time()
features = [18,0]
plot_partial_dependence(model, trainX, features, feature_names=column_names,fig= fig_4)
fig_4.savefig('result4.png')
print("Time consumed:", time.time() - time_start)


