#!/usr/bin/env python
# coding: utf-8

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
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import RandomOverSampler


data = pd.read_sas('/home/shared/cprhd/DATA_CPRHD_SES/wnv_2245new.sas7bdat') 
print("Checkpoint 1")

x_selected = data[data.drop(columns=['wnvbinary','yrweeks','yrwksfid','yr_hexid','year']).columns[[7,8,9,10,15,16,17,20]]]
y = data['wnvbinary'].values 
y_selected = y
print("Checkpoint 2")

def model_RF_test(model_RF, dataX, dataY):
    print("Model performance")
    predict_data = model_RF.predict_proba(dataX)
    
    # Some stats
    print("Feature Importantce : ")
    print(model_RF.feature_importances_)
    print("Total number of WNV occurence in test set : " + str(len(dataY[dataY > 0])))
    
    print("Number of WNV occurence the model is able to capture in test set:" + str(dataY[np.where(predict_data[:,1]  > 0)].sum()))
    
    print("Expected number of WNV occurence of models aussume prediction is normally distributed : " + str(predict_data[:,1].mean() * len(predict_data)))
    print("Log loss : " + str(log_loss(dataY,predict_data)))
    
    print("This is to test the performance of random forest model, ideally, the logloss is low and also it is able to capture most of the WNV occurence, and also it is neither over predicting " +  
          "nor under predicting")
    print("AUC : " + str( roc_auc_score(dataY,predict_data[:,1])))
    
    return None # Check how many wnv it predicts

print("Checkpoint 3")

trainX_sel, testX_sel, trainY_sel, testY_sel = train_test_split(x_selected.values, y_selected, test_size = 0.2, shuffle = True) # CV


rus = RandomOverSampler(random_state=0)
X_resampled, y_resampled = rus.fit_sample(trainX_sel,trainY_sel)
time_start = time.time()
model_RF1 = RandomForestClassifier(n_estimators=500,
                                 n_jobs = -1,
                               max_features='sqrt',
                                 max_depth= None,
                                 bootstrap=True,
                                   min_samples_leaf= 2,
                                   class_weight='balanced'
                                 ) # Use undersampling to see if it worked
model_RF1.fit(X_resampled, y_resampled)
print("Checkpoint 4 - time consumed:", time.time() - time_start) 


model_RF_test(model_RF1,testX_sel,testY_sel)






