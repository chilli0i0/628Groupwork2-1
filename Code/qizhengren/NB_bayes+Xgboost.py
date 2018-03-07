# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 19:09:36 2018

@author: rqz
"""

import pandas as pd
import string, re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import math
import time
start = time.time()
import pickle
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


################### training part ##########################3

df = pd.read_csv("train_translation.csv")
train = df.loc[:, ['stars', 'text']]
label_cols = ['1', '2', '3', '4', '5']
for i in label_cols:
    train[i] = (train['stars'] == int(i))
re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
def tokenize(s): return re_tok.sub(r' \1 ', s).split()

n = train.shape[0]

vec = TfidfVectorizer(ngram_range=(1,2), tokenizer=tokenize,
                      min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,
                      smooth_idf=1, sublinear_tf=1)
trn_term_doc = vec.fit_transform(train['text'])


pickle.dump(vec, open('./NBSVM_model/vectorizer.pk', 'wb'))

def pr(y_i, y):
    p = x[y == y_i].sum(0)
    return (p+1) / ((y==y_i).sum()+1)

x = trn_term_doc

def get_mdl(y):
    y = y.values
    r = np.log(pr(1,y) / pr(0,y))
    m = LogisticRegression(C=4, dual=True)
    x_nb = x.multiply(r)
    return m.fit(x_nb, y), r

for i, j in enumerate(label_cols):
    print('fit', j)
    m,r = get_mdl(train[j])
    m_name = './NBSVM_model/m' + j + '.sav'
    r_name = './NBSVM_model/r' + j
    pickle.dump(m, open(m_name, 'wb'))
    np.save(r_name, r)

################### prediction part for xgboost train ############################
df = pd.read_csv("train_translation.csv")
################### prediction part for xgboost test###############
df = pd.read_csv("test_translation_new.csv")
########then all the same######
test = df.loc[:, [ 'text']]

label_cols = ['1', '2', '3', '4', '5']

re_tok = re.compile(f'([{string.punctuation}“”¨«»´·º½¾¿¡§£₤‘’])')
def tokenize(s): return re_tok.sub(r' \1 ', s).split()

n = test.shape[0]

vec_name = './NBSVM_model/vectorizer.pk'
vec = pickle.load(open(vec_name, 'rb'))

test_term_doc = vec.transform(test['text'])

preds = np.zeros((test.shape[0], len(label_cols)))

for i, j in enumerate(label_cols):
    print('fit', j)
    m_name = './NBSVM_model/m' + j + '.sav'
    r_name = './NBSVM_model/r' + j + '.npy'
    m = pickle.load(open(m_name, 'rb'))
    r = np.asmatrix(np.load(r_name))
    preds[:,i] = m.predict_proba(test_term_doc.multiply(r))[:,1]
    
preds = np.asmatrix(preds)
stars = np.matrix([1, 2, 3, 4, 5]).transpose()
tmp_preds = preds.sum(1)
tmp_preds = preds / tmp_preds
tmp = tmp_preds * stars
aaa=pd.DataFrame(tmp_preds)
#create test set for xgboost##
aaa.to_csv('test_for_xgboost.csv',index=False,header=True,encoding='utf-8')
##create train set for xgboost##
train=pd.concat([aaa,df.iloc[:,0]],axis=1)
train.to_csv('train_for_xgboost.csv',index=False,header=True,encoding='utf-8')
### read train and test data
train=pd.read_csv('train_for_xgboost.csv')
test=pd.read_csv('test_for_xgboost.csv')
##train the data
x_train=train.iloc[:,:5]
target_train=train.iloc[:,-1]
import xgboost as xgb
xgb1=xgb.XGBRegressor(seed=1,max_depth=7,n_estimators=100)
xgb1.fit(x_train,target_train.values.ravel())
result1=xgb1.predict(test)
result=pd.DataFrame(result1)
result.columns=['result']
new_data=pd.DataFrame(columns=['Id','Prediction1'])
new_data['Id'] = range(1,len(result1)+1)
new_data['Prediction1']=list(result['result'])
new_data.to_csv('predict_new1.csv',index=False,encoding='utf-8')