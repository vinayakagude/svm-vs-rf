#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Aug 18 19:00:12 2016

@author: Vinayaka Gude
"""

# Comparing Support Vector Machines and Rnadom Forests for Sisporto 2.0 Dataset.

import csv
import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier 

#Loading the data
data = []
file  = open('toco.csv', "r")
read = csv.reader(file)
for row in read :
    data.append(row)


# Parse the data into inputs and targets from the data file(toco.csv). 
arraydata = np.array(data)
inputs = np.delete(arraydata,np.s_[32],1)
targets = np.delete(arraydata,np.s_[0:32],1)

# divide the data into training and testing data sets. 
inputs = inputs.astype(np.float)
outputs = targets.astype(np.float)
trainin, testin = inputs[:1063,:], inputs[1063:,:]
trainout, testout = outputs[:1063,:], outputs[1063:,:]

trainout = np.array(trainout,np.float32).reshape(len(trainout),1)
trainout = trainout.reshape(-1,1)
trainout = np.concatenate(trainout, axis=0 )
trainout = np.array(trainout)

# Random Forest
random_forest = RandomForestClassifier(n_estimators = 100,random_state=10000)
random_forest = random_forest.fit(trainin,trainout)
predict_rf    = random_forest.predict(testin)

# Support Vector Machine
Support_Vector_Machine = svm.SVC(gamma=0.04, C=4,degree=3, kernel='linear')
Support_Vector_Machine = Support_Vector_Machine.fit(trainin, trainout)
predict_svm            = Support_Vector_Machine.predict(testin)

#reshaping testout
testout = np.array(testout,np.float32).reshape(len(testout),1)
testout = testout.reshape(-1,1)
testout = np.concatenate(testout, axis=0 )
testout = np.array(testout)


accurate_rf =0
accurate_svm=0

for x in range(len(predict_rf)):
    if predict_rf[x]  == testout[x]:
        accurate_rf   =  accurate_rf+1
    if predict_svm[x] == testout[x]:
        accurate_svm  =  accurate_svm+1

print (accurate_rf)
print ('The accuracy for the RandomForest is ', accurate_rf/len(testout)*100)
print (accurate_svm)
print ('The accuracy for the RandomForest is ', accurate_svm/len(testout)*100)


# Identification of the most important features for prediction. 

# ExtraTreesClassifier for ranking the features used. 
from sklearn.ensemble import ExtraTreesClassifier

model = ExtraTreesClassifier()
model.fit(inputs, outputs)
print(model.feature_importances_)

# Logistic Regression to identify the 10 most important
# features contributing towards the prediction.
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
rfe   = RFE(model, 10)
fit   = rfe.fit(inputs, outputs)
print(fit.n_features_) 
print(fit.support_)
print(fit.ranking_) 

