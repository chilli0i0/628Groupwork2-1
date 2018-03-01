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


# TODO: try the methods with vader sentiment scores

# read trainDataVecs and train_centroids and merge them together
trainDataVecs = np.load('trainDataVecs.npy')
train_centroids = np.load('train_centroids.npy')

trainDataVecsDf = pd.DataFrame(trainDataVecs)
train_centroidsDf = pd.DataFrame(train_centroids)

frames = [trainDataVecsDf]
result = pd.concat(frames,axis=1)
#np.savetxt("x.csv", result, delimiter=",")
#df['stars'][0:50000].to_csv('target.csv', sep=',')
#result.to_pickle('reviews_information.csv')
#result = pd.read_pickle('reviews_information.csv')

x_train = np.float64(result.iloc[25000:45000,:].as_matrix())
target_train = df['stars'][25000:45000]

x_test = np.float64(result.iloc[45000:50000,:].as_matrix())
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
svc = SVR(C=1.5, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma='auto',
    kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
svc.fit(X_train,target_train.values.ravel())
X_test = StandardScaler().fit_transform(x_test)
result=svc.predict(X_test)
mse_calculation([float(i) for i in target_test],list(result))
#0.83

# xgboost


# GBDT
from sklearn.ensemble import GradientBoostingRegressor
gbdt = GradientBoostingRegressor(n_estimators=100,loss='ls',max_depth=5)
gbdt_fit = gbdt.fit( x_train, target_train.values.ravel() )
gbdt_predict = gbdt.predict( x_test )
mse_calculation([float(i) for i in target_test],list(gbdt_predict))
#0.85

# Naive Bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
X_train = StandardScaler().fit_transform(x_train)
X_test = StandardScaler().fit_transform(x_test)
gnb = GaussianNB(priors=None)
gnb_fit = gnb.fit(X_train,target_train.values.ravel())
gnb_predict = gnb.predict( X_test )
mse_calculation([float(i) for i in target_test],list(gnb_predict))
# 1.36



# Logistic regression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight="balanced", random_state=None, solver='saga', max_iter=100, multi_class='multinomial', verbose=0, warm_start=False, n_jobs=1)
lr_fit = lr.fit(x_train,target_train.values.ravel())
lr_predict = lr.predict( x_test )
mse_calculation([float(i) for i in target_test],list(lr_predict))
# 0.96


# Decision Tree
from sklearn.tree import DecisionTreeClassifier
DTC = DecisionTreeClassifier(max_depth=10)
DTC_fit = DTC.fit(x_train,target_train.values.ravel())
DTC_predict = DTC.predict( x_test )
mse_calculation([float(i) for i in target_test],list(DTC_predict))
# 1.21

from sklearn.tree import DecisionTreeRegressor
DTR = DecisionTreeRegressor(max_depth=7)
DTR_fit = DTR.fit(x_train,target_train.values.ravel())
DTR_predict = DTR.predict( x_test )
mse_calculation([float(i) for i in target_test],list(DTR_predict))
# 1.04

# K-nearest neighbor
from sklearn.neighbors import KNeighborsClassifier
KNNC = KNeighborsClassifier(n_neighbors=5,weights='distance', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=1)
KNNC_fit = KNNC.fit(x_train,target_train.values.ravel())
KNNC_predict = KNNC.predict( x_test )
mse_calculation([float(i) for i in target_test],list(KNNC_predict))
#1.20

from sklearn.neighbors import KNeighborsRegressor
KNNR = KNeighborsRegressor(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None,  n_jobs=1)
KNNR_fit = KNNR.fit(x_train,target_train.values.ravel())
KNNR_predict = KNNR.predict( x_test )
mse_calculation([float(i) for i in target_test],list(KNNR_predict))
#1.04
# linear regression
from sklearn.linear_model import LinearRegression
lm = LinearRegression(fit_intercept=True, normalize=True, copy_X=True, n_jobs=1)
lm_fit = lm.fit(x_train,target_train.values.ravel())
lm_predict = lm.predict( x_test )
mse_calculation([float(i) for i in target_test],list(lm_predict))
# 0.84


# linear regression - lasso
from sklearn.linear_model import Lasso
lml = Lasso(alpha=0.0001, fit_intercept=True, normalize=False, precompute=False, copy_X=True, max_iter=1000, tol=0.0001, warm_start=True, positive=False, random_state=None, selection='cyclic')
lml_fit = lml.fit(x_train,target_train.values.ravel())
lml_predict = lml.predict( x_test )
mse_calculation([float(i) for i in target_test],list(lml_predict))
# 0.86 with a very small alpha

# linear regression - redge
from sklearn.linear_model import Ridge
lmr = Ridge(alpha=0.00001, fit_intercept=True, normalize=False, copy_X=True, max_iter=None, tol=0.001, random_state=None)
lmr_fit = lmr.fit(x_train,target_train.values.ravel())
lmr_predict = lmr.predict( x_test )
mse_calculation([float(i) for i in target_test],list(lmr_predict))
# 0.84

# hard voting

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

clf1 = GradientBoostingRegressor()
clf2 = DecisionTreeClassifier()
clf3 = KNeighborsClassifier()
eclf = VotingClassifier(estimators=[('gbdt', clf1), ('lr', clf2), ('ridge', clf3)], voting='hard')
eclf_fit = eclf.fit( x_train, target_train.values.ravel() )
eclf_predict = eclf.predict( np.int32(x_test) )
mse_calculation([float(i) for i in target_test],list(eclf_predict))

# soft voting

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier



# bagging

from sklearn.ensemble import BaggingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
X_train = StandardScaler().fit_transform(x_train)
X_test = StandardScaler().fit_transform(x_test)
bagging = BaggingClassifier(GaussianNB(),max_samples=0.5, max_features=0.5)
bag_fit = bagging.fit(X_train,target_train.values.ravel())
bag_predict = bagging.predict( X_test )
mse_calculation([float(i) for i in target_test],list(bag_predict))
