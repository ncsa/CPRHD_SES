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

model = pickle.load(open('/home/shared/cprhd/MODELS/RF_model_max','rb'))
data = pd.read_sas('/home/shared/cprhd/DATA_CPRHD_SES/wnv_2245new.sas7bdat')
x = data.drop(columns=['wnvbinary','yrweeks','yrwksfid','yr_hexid','year']).values # Drop extra column
y = data['wnvbinary'].values
trainX, testX, trainY, testY = train_test_split(x, y, test_size = 0.2, random_state = 1)

time_start = time.time()
features = [46,6,15,10]
plot_partial_dependence(model, trainX[:100], features, feature_names=data.drop(columns=['wnvbinary','yrweeks','yrwksfid','yr_hexid','year']).columns)
print("Time consumed:", time.time() - time_start)

time_start = time.time()
features = [46,6,15,10]
plot_partial_dependence(model, trainX[:200], features, feature_names=data.drop(columns=['wnvbinary','yrweeks','yrwksfid','yr_hexid','year']).columns)
print("Time consumed:", time.time() - time_start)

time_start = time.time()
features = [19,20,21,22]
plot_partial_dependence(model, trainX[:100], features, feature_names=data.drop(columns=['wnvbinary','yrweeks','yrwksfid','yr_hexid','year']).columns)
print("Time consumed:", time.time() - time_start)

time_start = time.time()
features = [19,20,21,22]
plot_partial_dependence(model, trainX[:100], features, feature_names=data.drop(columns=['wnvbinary','yrweeks','yrwksfid','yr_hexid','year']).columns)
print("Time consumed:", time.time() - time_start)

time_start = time.time()
features = [19,20,21,22]
plot_partial_dependence(model, trainX[:1000], features, feature_names=data.drop(columns=['wnvbinary','yrweeks','yrwksfid','yr_hexid','year']).columns)
print("Time consumed:", time.time() - time_start)

time_start = time.time()
features = [(46,10),(20,22)]
plot_partial_dependence(model, trainX[:500], features, feature_names=data.drop(columns=['wnvbinary','yrweeks','yrwksfid','yr_hexid','year']).columns)
print("Time consumed:", time.time() - time_start)
