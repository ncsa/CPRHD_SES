import numpy as np
import time
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import log_loss
import pickle
from sklearn.utils import class_weight

time_start = time.time()
data = pd.read_sas('/home/shared/cprhd/DATA_CPRHD_SES/wnv_2245new.sas7bdat') # In the Cook_Dupage Directory
print("Data read in:", time.time() - time_start)

time_start = time.time()
x = data.drop(columns=['wnvbinary','yrweeks','yrwksfid','yr_hexid','year','income1'])

x_selected = x.drop(columns=x.columns[[4,5,25,26,-17,-16,-15,-14,-13,-12,-11,-10,-9,-8,-6]]).values
y_selected = data['wnvbinary'].values
print("Data selected in:", time.time() - time_start)
time_start = time.time()
trainX_sel, testX_sel, trainY_sel, testY_sel = train_test_split(x_selected, y_selected, test_size = 0.2, random_state=1) # CV
print("data split:", time.time() - time_start)
class_weights = class_weight.compute_class_weight('balanced' , classes=np.unique(y_selected),y = y_selected)
time_start = time.time()
model_RF_best_2 = RandomForestClassifier(n_estimators=1000,
                                 n_jobs = -1,
                                 max_features=None,
                                 max_depth=None,
                                 bootstrap=True,
                                 min_samples_leaf=2
                                 )
print("Classifier established:", time.time() - time_start)


time_start = time.time()
model_RF_best_2.fit(trainX_sel, trainY_sel)
print("model fit:", time.time() - time_start)




time_start = time.time()

class_weights = {0: class_weights[0], 1: class_weights[1]}

param_grid = {
    'bootstrap': [True],
    'max_depth': [60,100],
    'max_features': [4,10],
    'min_samples_leaf': [5,8],
    'min_samples_split': [5,8],
    'n_estimators': [1000],
    'class_weight': [class_weights,None]
}


CV_model_RF_3 = GridSearchCV(model_RF_best_2, param_grid, scoring='neg_log_loss', cv=5)
print("CV model parameterized:", time.time() - time_start)

time_start = time.time()
CV_model_RF_3.fit(trainX_sel, trainY_sel)
print("CV model fit:", time.time() - time_start)

pickle.dump(CV_model_RF_3, open('RF_model', 'wb'))

def model_RF_test(model_RF, dataX, dataY):
    print("Model performance")
    predict_data = model_RF.predict_proba(dataX)
    # Some stats
    print("Feature Importance : ")
    print(model_RF.best_estimator_.feature_importances_)
    print("Total number of WNV occurrence in test set : " + str(len(dataY[dataY > 0])))

    print("Number of WNV occurrence the model is able to capture in test set:" + str(
        dataY[np.where(predict_data[:, 1] > 0)].sum()))

    print("Log loss : " + str(log_loss(dataY, predict_data)))

    print(
        "This is to test the performance of random forest model, ideally, the logloss is low and also it is able to "
        "capture most of the WNV occurrence")

    return None  # Check how many wnv it predicts


model_RF_test(CV_model_RF_3,testX_sel,testY_sel)
