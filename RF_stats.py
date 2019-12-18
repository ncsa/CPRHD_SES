import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score

"""
This script generates the stats of the model created in RF_model.py
"""

def model_RF_test(model_RF, dataX, dataY, table):
    print("Model performance")
    predict_data = model_RF.predict_proba(dataX)

    # Some stats
    print("Feature Importance : ")
    feature_importances = pd.DataFrame(model_RF.feature_importances_, index = table.columns, columns=['importance'])
    print(model_RF.best_estimator_.feature_importances_)
    print("Total number of WNV occurrence in test set : " + str(len(dataY[dataY > 0])))

    print("Number of WNV occurrence the model is able to capture in test set:" + str(
        dataY[np.where(predict_data[:, 1] > 0)].sum()))

    print("Log loss : " + str(log_loss(dataY, predict_data)))

    print("AUC: " + str(roc_auc_score(dataY, predict_data[:,1])))

    print(
        "This is to test the performance of random forest model, ideally, the logloss is low and also it is able to "
        "capture most of the WNV occurrence")

    return None  # Check how many wnv it predicts

data = pd.read_sas('/home/shared/cprhd/DATA_CPRHD_SES/wnv_2245new.sas7bdat') # In the Cook_Dupage Directory

x = data.drop(columns=['wnvbinary', 'yrweeks', 'yrwksfid', 'yr_hexid', 'year', 'income1'])
x_test = data[data.drop(columns=['wnvbinary','yrweeks','yrwksfid','yr_hexid','year']).columns[:]]

x_selected = x[x.drop(columns=x_test.columns[[4, 5, 25, 26, -17, -16, -15, -14, -13, -12, -11, -10, -9, -8, -6]]).columns[[2,4,5,6,7,12,13,14,15,16,17]]].values
y_selected = data['wnvbinary'].values


trainX_sel, testX_sel, trainY_sel, testY_sel = train_test_split(x_selected, y_selected, test_size = 0.2, random_state  = 1) # CV

loaded_model = pickle.load(open('/home/jallen17/CPRHD_SES/RF_agg_model_CV_final', 'rb'))
model_RF_test(loaded_model, testX_sel, testY_sel, x_selected)
