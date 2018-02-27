#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 22:19:11 2018

@author: hanmoli
"""
import pandas as pd
import numpy as np
import math
df=pd.read_csv("train_data.csv")
variables = pd.read_pickle('variables')

def mse_calculation(target,prediction):
    prediction_restrain = list()
    for i in prediction:
        if i>=1 and i<=5:
            prediction_restrain.append(i)
        elif i<1:
            prediction_restrain.append(1)
        else:
            prediction_restrain.append(5)
    return math.sqrt(sum(map(lambda x,y:np.square(x-y),target,prediction_restrain))/len(target))




# read trainDataVecs and train_centroids and merge them together
trainDataVecs = np.load('trainDataVecs.npy')
train_centroids = np.load('train_centroids.npy')

trainDataVecsDf = pd.DataFrame(trainDataVecs)
train_centroidsDf = pd.DataFrame(train_centroids)

frames = [trainDataVecsDf,variables]
result = pd.concat(frames,axis=1)
np.savetxt("x.csv", result, delimiter=",")
df['stars'][0:50000].to_csv('target.csv', sep=',')
#result.to_pickle('reviews_information.csv')
#result = pd.read_pickle('reviews_information.csv')

x_train = result.iloc[25000:45000,:].as_matrix()
target_train = df['stars'][25000:45000]

x_test = result.iloc[45000:50000,:].as_matrix()
target_test = df['stars'][45000:50000]

# random forest
from sklearn.ensemble import RandomForestRegressor
forest = RandomForestRegressor(n_estimators = 100) 
forest = forest.fit( x_train, target_train.values.ravel() )    
predict = forest.predict( x_test )    
mse_calculation([float(i) for i in target_test],list(predict))
# 0.90


# SVM
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
X_train = StandardScaler().fit_transform(x_train)
svc = SVR()
svc.fit(X_train,target_train.values.ravel())
X_test = StandardScaler().fit_transform(x_test)
result=svc.predict(X_test)
mse_calculation([float(i) for i in target_test],list(result))
#0.83

# xgboost


# GBDT
from sklearn.ensemble import GradientBoostingRegressor
gbdt = GradientBoostingRegressor(n_estimators=100) 
gbdt_fit = gbdt.fit( x_train, target_train.values.ravel() )    
gbdt_predict = gbdt.predict( x_test )    
mse_calculation([float(i) for i in target_test],list(gbdt_predict))
#0.86

# Naive Bayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb_fit = gnb.fit(x_train,target_train.values.ravel())
gnb_predict = gnb.predict( x_test )    
mse_calculation([float(i) for i in target_test],list(gnb_predict))
# 1.36



# Logistic regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(random_state=1)
lr_fit = lr.fit(x_train,target_train.values.ravel())
lr_predict = lr.predict( x_test )    
mse_calculation([float(i) for i in target_test],list(lr_predict))
# 0.96


# Decision Tree
from sklearn.tree import DecisionTreeClassifier
DTC = DecisionTreeClassifier()
DTC_fit = DTC.fit(x_train,target_train.values.ravel())
DTC_predict = DTC.predict( x_test )    
mse_calculation([float(i) for i in target_test],list(DTC_predict))
# 1.33

# K-nearest neighbor


# linear regression - lasso

# linear regression - redge


# hard voting

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier

clf1 = LogisticRegression(random_state=1)
clf2 = RandomForestClassifier(random_state=1)
clf3 = GaussianNB()
eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')
x = result.iloc[25000:45000,:].as_matrix()
target = pd.DataFrame(df['stars'][25000:45000],columns=['stars'])
gbdt_fit = eclf.fit( x, target.values.ravel() )    
gbdt_fit = eclf.predict( result.iloc[45000:50000,:].as_matrix() )    
mse_calculation([float(i) for i in df['stars'][45000:50000]],list(gbdt_fit))

# soft voting

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier

