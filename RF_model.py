#!/usr/bin/env python
# coding: utf-8

# In[2]:


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


# # Explore and extra data of particular periods of interest

# In[3]:


data = pd.read_sas('/home/guangya/Downloads/wnv_2245new.sas7bdat') #Data from week 22 to 45, which is what i used for latter models


# In[4]:


data.head() # Available features in the data set, More description is the final.xlsw file


# In[5]:


data.isna().sum() # Check na


# In[6]:


data.drop(columns=['wnvbinary','yrweeks','yrwksfid','yr_hexid','year']).isna().sum() # Drop year column so that so na appear


# In[7]:


x_total = data.drop(columns=['wnvbinary','yrweeks','yrwksfid','yr_hexid','year']).values # Drop extra column
y_total = data['wnvbinary'].values


# In[8]:


x_total = x_total.astype('float64')


# In[9]:


# some spot check for data
data[data['hexid'] == 1431]['blackpct'].unique()


# In[10]:


data[data['hexid'] == 1831]['whitepct'].unique()


# In[11]:


data[data['hexid'] == 1831]['dmipct'].unique() 


# In[12]:


data[data['hexid'] == 3121]['income1'].unique() # The Geological and social data is likely a 10 year estimate here, which does not change from 2005-2016


# In[13]:


x = data[['yr','templag2','templag3','templag4','precilag2','mirlag1','mirlag2','mirlag3','mirlag4', 'whitepct','owpct','dmipct','dhipct']].values
# Data set for the best model described in paper, table5.
# However, random forest use a different feature selection algorithm 
# so that this might not be the optimal one for oue models. Since it's much slower to train all, I will use ALL features for further optimization later
y = data['wnvbinary'].values 
x = x.astype('float64')


# In[14]:


trainX, testX, trainY, testY = train_test_split(x, y, test_size = 0.2, shuffle = True) # CV


# In[15]:


trainX_total, testX_total, trainY_total, testY_total = train_test_split(x_total, y_total, test_size = 0.2, shuffle = True) # CV for all


# # Test on Random Forest

# In[59]:


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
    
    print("This is to test the performance of random forest model, ideally, the logloss is low and also it is able to capture most of the WNV occurence, and also it is neither over predicting" +  
          "nor under predicting")
    
    return None # Check how many wnv it predicts


# In[17]:


from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(random_state=0)
X_resampled, y_resampled = rus.fit_resample(trainX_total, trainY_total)
time_start = time.time()
model_RF1 = RandomForestClassifier(n_estimators=400,
                                 n_jobs = -1,
                                 max_features=None,
                                 max_depth= None,
                                 bootstrap=True,
                                class_weight='balanced'
                                 ) # Use undersampling to see if it worked
model_RF1.fit(X_resampled, y_resampled)
print("time consumed:", time.time() - time_start) 


# In[60]:


model_RF_test(model_RF1,testX_total,testY_total) # The result tend to predict a lot of 1s, which is very biased.


# In[19]:


rus = RandomUnderSampler(random_state=0)
X_resampled, y_resampled = rus.fit_resample(trainX, trainY)
time_start = time.time()
model_RF2 = RandomForestClassifier(n_estimators=400,
                                 n_jobs = -1,
                                 max_features=None,
                                 max_depth= None,
                                 bootstrap=True,
                                class_weight='balanced'
                                 ) # Use undersampling to see if it worked
model_RF2.fit(X_resampled, y_resampled)
print("time consumed:", time.time() - time_start) 


# In[61]:


model_RF_test(model_RF2,testX,testY) 


# In[33]:


time_start = time.time()
model_RF3 = RandomForestClassifier(n_estimators=200,
                                 n_jobs = -1,
                                 max_features='sqrt',
                                 max_depth= 20,
                                 bootstrap=True,
                                   min_samples_leaf= 2,
                                class_weight='balanced'
                                 ) # Try more trees and see if calibration curve gets better
model_RF3.fit(trainX_total, trainY_total)
print("time consumed:", time.time() - time_start) 


# In[88]:


x = sorted(model_RF3.feature_importances_)[::-1]
seq = [list(model_RF3.feature_importances_).index(v) for v in x]


# In[91]:


seq[:10]


# In[57]:


model_RF_test(model_RF3,testX_total,testY_total)


# In[102]:


data.drop(columns=['wnvbinary','yrweeks','yrwksfid','yr_hexid','year']).columns


# In[35]:


time_start = time.time()
model_RF4 = RandomForestClassifier(n_estimators=200,
                                 n_jobs = -1,
                                 max_features='sqrt',
                                 max_depth= 20,
                                 bootstrap=True,
                                   min_samples_leaf= 2,
                                class_weight='balanced'
                                 ) # Try more trees and see if calibration curve gets better
model_RF4.fit(trainX, trainY)
print("time consumed:", time.time() - time_start) 


# In[62]:


model_RF_test(model_RF4,testX,testY)


# In[105]:


data[['yr','templag2','templag3','templag4','precilag2','mirlag1','mirlag2','mirlag3','mirlag4', 'whitepct','owpct','dmipct','dhipct']].columns


# In[104]:


data.drop(columns=['wnvbinary','yrweeks','yrwksfid','yr_hexid','year']).columns[[17,20,9,10,16,18]]West Nile Virus Chicago: Wayne


# In[96]:


data[data.drop(columns=['wnvbinary','yrweeks','yrwksfid','yr_hexid','year']).columns[[7,8,9,10,15,16,17,20]]].head() # From the above 3 feature importance of models, we can manually select 
# about 7 features which is mostly important


# In[97]:


x_selected = data[data.drop(columns=['wnvbinary','yrweeks','yrwksfid','yr_hexid','year']).columns[[7,8,9,10,15,16,17,20]]]
y_selected = y


# ### From the above model, we can see that random forest, although is not very good, can actually capture someinformation. So we will try formal Cross validation on selected features and see if it gets betterz

# ## Pipnelines for later for wrok

# In[98]:


trainX_sel, testX_sel, trainY_sel, testY_sel = train_test_split(x_selected.values, y_selected, test_size = 0.2, shuffle = True) # CV


# ## Find best model 1

# In[39]:


time_start = time.time()
params_RF_grid_1 = {
    'n_estimators' : [500, 1000],
    'max_features' : [90, 'sqrt', None],
    'max_depth' : [10, None],
    'min_samples_leaf' : [1,2]
}
CV_model_RF_1 = GridSearchCV(model_RF, params_RF_grid_1, scoring='neg_log_loss',cv=5)
CV_model_RF_1.fit(x_selected, dataY)
print("time consumed:", time.time() - time_start)


# In[40]:


CV_model_RF_1.best_estimator_


# In[41]:


CV_model_RF_1.best_params_


# In[99]:


time_start = time.time()
model_RF_best_1 = RandomForestClassifier(n_estimators=800,
                                 n_jobs = -1,
                                 max_features="sqrt",
                                 max_depth=None,
                                 bootstrap=True,
                                 min_samples_leaf=2
                                 )
model_RF_best_1.fit(trainX_sel,trainY_sel)
print("Time consumed:", time.time() - time_start)


# In[100]:


model_RF_test(model_RF_best_1,testX_sel,testY_sel)


# ## Find best model 2

# In[ ]:


time_start = time.time()
params_RF_grid_2 = {
    'n_estimators' : [800, 1200],
    'max_features' : ['sqrt', 5],
    'min_samples_leaf' : [2,3]
}
CV_model_RF_2 = GridSearchCV(model_RF, params_RF_grid_2, cv=5)
CV_model_RF_2.fit(dataX, dataY)
print("time consumed:", time.time() - time_start)


# In[24]:


CV_model_RF_2.fit(dataX, dataY)


# In[ ]:


CV_model_RF_2.best_params_


# In[11]:


time_start = time.time()
model_RF_best_2 = RandomForestRegressor(n_estimators=1500,
                                 criterion="mse",
                                 n_jobs = -1,
                                 max_features=None,
                                 max_depth=None,
                                 bootstrap=True,
                                 min_samples_leaf=2
                                 )
model_RF_best_2.fit(trainX, trainY)
print("Time consumed:", time.time() - time_start)


# ## Find best model 3

# In[12]:


time_start = time.time()
param_grid = {
    'bootstrap': [True],
    'max_depth': [80, None, 110],
    'max_features': ['sqrt', 4],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [500, 2000, 4000, 1000]
}
CV_model_RF_3 = GridSearchCV(model_RF_best_2, params_RF_grid_3, cv=5)
CV_model_RF_3.fit(dataX, dataY)
print("time consumed:", time.time() - time_start)


# In[13]:


CV_model_RF_3.best_estimator_


# In[14]:


CV_model_RF_3.best_params_


# In[51]:


time_start = time.time()
model_RF_best_3 = RandomForestRegressor(n_estimators=8000,
                                 criterion="mse",
                                 n_jobs = -1,
                                 max_features="log2",
                                 max_depth=None,
                                 bootstrap=True,
                                 min_samples_leaf=2
                                 )
model_RF_best_3.fit(trainX, trainY)
print("Time consumed:", time.time() - time_start)

