# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 19:28:17 2018

@author: rqz
"""
train_sample=train.iloc[:300000,:]
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
param_test1 = {
 'subsample':[i/100.0 for i in range(55,70,5)],
 'colsample_bytree':[i/100.0 for i in range(75,90,5)]
}
param_test1 = {
 'reg_alpha':[0, 0.001, 0.005, 0.01, 0.05],
 'learning_rate':[0.01,0.05,0.1,0.15]
}
gsearch1=GridSearchCV(estimator=XGBRegressor(
        base_score=0.5,
        colsample_bylevel=1,
        subsample=0.6,
        colsample_bytree=0.8,
        gamma=0.3,
        max_delta_step=0,
        missing=None,
        n_estimators=100,
        nthread=-1,
        reg_lambda=1,
        max_depth=4,
        min_child_weight=6,
        scale_pos_weight=1,
        seed=1),param_grid=param_test1,scoring='neg_mean_squared_error',cv=2)
gsearch1.fit(train_sample.iloc[:,:5],train_sample.iloc[:,-1])
gsearch1.grid_scores_,gsearch1.best_params_,gsearch1.best_score_
import numpy as np
import math
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


test_sample=train.iloc[300000:400000,:5]



xgb1=xgb.XGBRegressor(seed=1,max_depth=4,n_estimators=100,
                      min_child_depth=6,gamma=0.3,colsample_bytree= 0.8,
                      subsample=0.6,learning_rate=0.15,reg_alpha=0.05)
xgb1.fit(train_sample.iloc[:,:5],train_sample.iloc[:,-1])
result1=xgb1.predict(test_sample)
mse_calculation(train.iloc[300000:400000,-1],result1)




#0.4827496244176916

