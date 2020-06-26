# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 18:57:04 2020

@author: Chandra mouli
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

tf.__version__#prints the version
dataset=pd.read_excel('Folds5x2_pp.xlsx')
x=dataset.iloc[:,:-1]
y=dataset.iloc[:,-1]#predict energy output

#train test split
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

#building the ANN
ann=tf.keras.models.Sequential()
#adding first hidden layer and ip layer
ann.add(tf.keras.layers.Dense(units=6,activation='relu'))

ann.add(tf.keras.layers.Dense(units=6,activation='relu'))

##op layer
ann.add(tf.keras.layers.Dense(units=1))

ann.compile(optimizer='adam',loss='mean_squared_error')

#Training ANN
ann.fit(x_train,y_train,batch_size=32,epochs=100)

#predicting
y_pred=ann.predict(x_test)
y_pred=y_pred.round(2)#decimal roundin off to 2

from sklearn.metrics import mean_absolute_error,mean_squared_error
mean_squared_error(y_test,y_pred)
