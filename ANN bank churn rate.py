# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 18:49:13 2020

@author: Chandra mouli
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

tf.__version__#prints the version

#data preprocessing
dataset=pd.read_csv("Churn_Modelling.csv")
x=dataset.iloc[:,3:-1]
y=dataset.iloc[:,-1]

#encoding categorical columns
dataset.info()#no null values
x1=x.iloc[:,1:3]
x2=pd.get_dummies(x1)
print(x1)
x2.info()


x=x.drop(['Geography','Gender'],1)
x=pd.concat([x,x2],axis=1)
#train test split
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

#feature scaling for all data's
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

#Building ANN
ann=tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=6,activation='relu'))


#second hidden layer
ann.add(tf.keras.layers.Dense(units=6,activation='relu'))

#op layer
ann.add(tf.keras.layers.Dense(units=1,activation='sigmoid'))


##Training the ANN
ann.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])#compiling the ANN#compling with optimizer,lossfn and metrics for finding errors


#training the ann on the training set
ann.fit(x_train,y_train,batch_size=32,epochs=100)

a=sc.transform([[600,40,3,60000,2,1,1,50000,1,0,0,0,1]])
ann.predict(a)
print(ann.predict(a)>0.5)
#final predic
y_pred=ann.predict(x_test)
y_pred=(y_pred>0.5)#if >0.5 its 1 that is true he will leave the bank
y_pred=y_pred.astype(int) #to convert to 0 and 1

#making confusion matrix
from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(y_test,y_pred)
accuracy_score(y_test,y_pred)

