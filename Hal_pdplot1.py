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

model = pickle.load(open('/home/shared/cprhd/MODELS/model_best_all','rb')) # Change this line to new model -- RF_model_CV_final_11-27
data = pd.read_sas('/home/shared/cprhd/DATA_CPRHD_SES/wnv_2245new.sas7bdat')
x = data.drop(columns=['wnvbinary', 'yrweeks', 'yrwksfid', 'yr_hexid', 'year', 'income1'])
x_selected_all = x.drop(columns=x.columns[[4, 5, 25, 26, -17, -16, -15, -14, -13, -12, -11, -10, -9, -8, -6]]).values
y = data['wnvbinary'].values
trainX, testX, trainY, testY = train_test_split(x_selected_all, y, test_size = 0.2, random_state = 1)

fig_1 = plt.figure()
time_start = time.time()
features = [5,9]
column_names = x.drop(columns=x.columns[[4, 5, 25, 26, -17, -16, -15, -14, -13, -12, -11, -10, -9, -8, -6]]).columns

plot_partial_dependence(model, trainX, features, feature_names=column_names,fig= fig_1)
fig_1.savefig('result1.png')
print("Time consumed:", time.time() - time_start)

fig_2 = plt.figure()
time_start = time.time()
features = [8,13]
plot_partial_dependence(model, trainX, features, feature_names=column_names,fig= fig_2)
fig_2.savefig('result2.png')
print("Time consumed:", time.time() - time_start)

fig_3 = plt.figure()
time_start = time.time()
features = [17,22]
plot_partial_dependence(model, trainX, features, feature_names=column_names,fig= fig_3)
fig_3.savefig('result3.png')
print("Time consumed:", time.time() - time_start)

fig_4 = plt.figure()
time_start = time.time()
features = [21,20]
plot_partial_dependence(model, trainX, features, feature_names=column_names,fig= fig_4)
fig_4.savefig('result4.png')
print("Time consumed:", time.time() - time_start)

fig_5 = plt.figure()
time_start = time.time()
features = [18,19]
plot_partial_dependence(model, trainX, features, feature_names=column_names,fig= fig_5)
fig_5.savefig('result5.png')
print("Time consumed:", time.time() - time_start)
