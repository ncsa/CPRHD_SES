#!/usr/bin/env python
# coding: utf-8

# In[38]:


import numpy as np
import keras
import scipy.stats as stats
import pylab as pl
import time
import json
import pandas as pd
from matplotlib import pyplot as plt
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from keras.layers import Dense, Dropout

from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn import metrics


# In[2]:


origin_data = np.loadtxt("/home/jallen17/DATA_CPRHD_SES/python_model_data/data.txt")


# In[3]:


origin_data2 = pd.read_csv("/home/jallen17/DATA_CPRHD_SES/R_model_data/selected_features_remove_outliers.csv")
origin_data2 = origin_data2.values


# In[4]:


data = origin_data
data.shape


# In[5]:


dataX = data[:,0:-1]
dataY = data[:,-1]
dataY = dataY.reshape(-1,1)


# # Some stats

# In[6]:


sorted_mir = -np.sort(-origin_data[:,-1])
print("max:", max(sorted_mir),"\nmin:", min(sorted_mir), "\nmean:", sorted_mir.mean(), "\nstd:", sorted_mir.std())
fit_mir = stats.norm.pdf(sorted_mir, np.mean(sorted_mir), np.std(sorted_mir))
pl.plot(sorted_mir, fit_mir, '-o')
pl.hist(sorted_mir, density=True)
pl.show()


# In[7]:


plt.plot(dataY, "*")


# In[8]:


mir_zscore = np.abs(stats.zscore(dataY))
print(mir_zscore)
outliers_zscore = np.where(mir_zscore > 3)
print(outliers_zscore)
print(len(outliers_zscore[0]))


# In[9]:


dataX = origin_data[:,0:-1]
dataY = origin_data[:,-1]
dataY = dataY.reshape(-1,1)


# In[10]:


dataX_drop = np.delete(dataX, outliers_zscore[0], axis=0)
dataY_drop = np.delete(dataY, outliers_zscore[0], axis=0)


# In[11]:


dataX = dataX_drop
dataY = dataY_drop


# In[12]:


dataX.shape


# In[13]:


sorted_mir = -np.sort(-dataY[:,0])
print("max:", max(sorted_mir),"\nmin:", min(sorted_mir), "\nmean:", sorted_mir.mean(), "\nstd:", sorted_mir.std())
fit_mir = stats.norm.pdf(sorted_mir, np.mean(sorted_mir), np.std(sorted_mir))
pl.plot(sorted_mir, fit_mir, '-o')
pl.hist(sorted_mir, density=True)
pl.show()


# In[14]:


plt.plot(dataY, "*")


# In[15]:


np.max(dataY)


# # Neural Network

# In[27]:


# generate training set and test set
trainX, testX, trainY, testY = train_test_split(dataX, dataY, test_size = 0.33, shuffle = True)


# In[23]:


def data_scale(trainX, testX, trainY, testY, scale_type = "z-score"):
    scaler_trainX = None
    scaler_trainY = None
    
    if scale_type == "z-score":
        scaler_trainX = preprocessing.StandardScaler().fit(trainX)
        trainX_scaled = scaler_trainX.transform(trainX)
        testX_scaled = scaler_trainX.transform(testX)

        scaler_trainY = preprocessing.StandardScaler().fit(trainY)
        trainY_scaled = scaler_trainY.transform(trainY)
        testY_scaled = scaler_trainY.transform(testY)
    
    elif scale_type == "min-max":
        scaler_trainX = preprocessing.MinMaxScaler().fit(trainX)
        trainX_scaled = scaler_trainX.transform(trainX)
        testX_scaled = scaler_trainX.transform(testX)

        scaler_trainY = preprocessing.MinMaxScaler().fit(trainY)
        trainY_scaled = scaler_trainY.transform(trainY)
        testY_scaled = scaler_trainY.transform(testY)
        
    return scaler_trainX, scaler_trainY, trainX_scaled, testX_scaled, trainY_scaled, testY_scaled
        


# In[29]:


# Scaling (Normalization)
# z-score (z = (x-u)/s)
scaler_trainX, scaler_trainY, trainX_scaled, testX_scaled, trainY_scaled, testY_scaled = data_scale(
    trainX, testX, trainY, testY, scale_type="z-score")
# min-max (x-x_min)/(x_max-x_min) * (max-min) + min
#scaler_trainX, scaler_trainY, trainX_scaled, testX_scaled, trainY_scaled, testY_scaled = data_scale(
#    trainX, testX, trainY, testY, scale_type="z-score")


# In[30]:


np.min(testY_scaled)


# In[24]:


def reconstruct_from_sclae(scaler, data):
    return scaler.inverse_transform(data)


# In[25]:


def model_NN_test(model_NN, dataX, dataY, scaler_trainX, scaler_trainY):
    print("Model performance")
    predict_data = model_NN.predict(dataX)
    
    if scaler_trainX != None and scaler_trainY != None:
        dataX = reconstruct_from_sclae(scaler_trainX, dataX)
        dataY = reconstruct_from_sclae(scaler_trainY, dataY)
        predict_data = reconstruct_from_sclae(scaler_trainY, predict_data)
    
    # Some stats
    print("MSE:", metrics.mean_squared_error(dataY, predict_data))
    print("MAE:", metrics.mean_absolute_error(dataY, predict_data))
    print("R2:", metrics.r2_score(dataY, predict_data))
    errors = abs((dataY - predict_data) / dataY)
    mean_errors = np.mean(errors)
    mean_accuracy = 1 - mean_errors
    print("Mean Accuracy:", mean_accuracy * 100, "%")
    
    plt.figure(1)
    plt.xlabel('True MIR')
    plt.ylabel('Predicted MIR')
    plt.plot(dataY, predict_data, "*")
    plt.show()
    
    plt.figure(2)
    plt.plot(dataY, label = 'actual data')
    plt.plot(predict_data, label = 'predict data')
    plt.xlabel('Tract')
    plt.ylabel('MIR')
    plt.legend(loc = 'best')
    plt.show()
    
    return errors


# In[26]:


def NN_model():
    dropout_rate = 0
    
    model = Sequential()
    model.add(Dense(256, activation = "elu", input_dim = 180))
    model.add(Dropout(dropout_rate))
    model.add(Dense(128, activation = "elu"))
    model.add(Dropout(dropout_rate))
    model.add(Dense(64, activation = "elu"))
    model.add(Dropout(dropout_rate))
    model.add(Dense(32, activation = "elu"))
    model.add(Dropout(dropout_rate))
    model.add(Dense(16, activation = "elu"))
    model.add(Dropout(dropout_rate))
    model.add(Dense(8, activation = "elu"))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, kernel_initializer='normal'))
    
    model.compile(loss = "mean_squared_error", optimizer = "adam") 
    return model


# In[35]:


seed = 0
np.random.seed(seed)
estimators = []
estimators.append(('standardize', preprocessing.StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=NN_model, epochs = 100, batch_size = 32)))
pipeline = Pipeline(estimators) 
kfold = KFold(n_splits=3, random_state=seed)
results = cross_val_score(estimators, dataX, dataY, cv=kfold)


# In[25]:


results.mean()


# In[71]:


time_start = time.time()
dropout_rate = 0.4
activation_function = "elu" # elu, relu, sigmoid, tanh, linear, softmax
    
model = Sequential()
model.add(Dense(16, kernel_initializer='normal', activation = "elu", input_dim = trainX_scaled.shape[1]))
model.add(Dropout(dropout_rate))
model.add(Dense(8, kernel_initializer='normal', activation = "elu"))
model.add(Dropout(dropout_rate))
model.add(Dense(1, kernel_initializer='normal'))

model.compile(loss = "mean_squared_error", optimizer = "adam") 
# mean_absolute_error, mean_squared_error;
# SGD, adam, RMSprop, Adagrad, Adadelta, Adamax, Nadam
train_history = model.fit(x = trainX_scaled, y = trainY_scaled, 
                                    epochs = 500, batch_size = 16, 
                                    validation_data = (testX_scaled, testY_scaled))
model_history = train_history.history
time_consumed = time.time() - time_start


# In[59]:


time_consumed


# In[60]:


errors_train = model_NN_test(model, trainX_scaled, trainY_scaled, scaler_trainX, scaler_trainY)


# In[61]:


errors_test = model_NN_test(model, testX_scaled, testY_scaled, scaler_trainX, scaler_trainY)


# In[65]:


plt.plot(model_history['loss'], label = 'loss')
plt.plot(model_history['val_loss'], label = 'val_loss')
plt.legend(loc = "best")


# In[32]:


len(errors_test)


# In[33]:


len(errors_test[errors_test<0.5])


# In[66]:


# save history
with open("model_500_history_2.json", "w") as history_file:
    history_file.write(json.dumps(train_history.history))


# In[67]:


# read history
with open("model_500_history_2.json") as history_file:
        model_history = json.loads(history_file.read())


# In[68]:


# save model
model.save("model_500_2.h5")


# In[69]:


# read model
model = keras.models.load_model('model_500_2.h5')


# In[70]:


model.summary()


# In[ ]:


def NN():
    trainX = None
    trainY = None
    testX = None
    testY = None
    model = None
    score = None
    
    def __init__ (self,
                  dropout_layer_rate = 0.1,
                  rnn_dropout_rate = 0.1,
                  nb_epoch = 50,
                  batch_size = 16,
                  loss = 'mean_absolute_error',
                  optimizer = 'adam',
                  save_model = True,
                  save_model_path = ''):

        self.dropout_layer_rate = dropout_layer_rate
        self.rnn_dropout_rate = rnn_dropout_rate
        self.nb_epoch = nb_epoch
        self.batch_size = batch_size
        self.loss = loss
        self.optimizer = optimizer #rmsprop, adam
        self.save_model = save_model
        self.save_model_path = save_model_path
        
    def NN_getData(self, trainX, trainY, testX, testY):
        self.trainX = trainX
        self.trainY = trainY
        self.testX = testX
        self.testY = testY

        return True
    
    def NN_model_train(self):
        trainX = self.trainX
        trainY = self.trainY
        testX = self.testX
        testY = self.testY
        model = self.model
        dropout_layer_rate = self.dropout_layer_rate
        rnn_dropout_rate = self.rnn_dropout_rate
        
        input_dim = trainX.shape
        
        model = Sequential()
        model.add()

