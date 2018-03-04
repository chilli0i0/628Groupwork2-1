import pandas as pd
# from rake_nltk import Rake
# import nltk
import string, re
# import collections
# import vaderSentiment
import random
# import nltk.sentiment
# from nltk.corpus import sentiwordnet as swn
# from sklearn import linear_model, metrics, datasets
import numpy as np
# import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import math
import time

start = time.time()

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

#train = pd.read_csv("train_new.csv")
#test = pd.read_csv("test_new.csv")

#random.seed(8102)
# select train and test samples
#sample_size = 346378
#test_size = 100000
#
#sample = random.sample(range(df.shape[0]), sample_size)
#
#test_sample = sample[99999:]
#sample = sample[0:99999]

# data insight
# lens = [len(x.split()) for x in df.text]
# lens = pd.Series(lens)
# df.loc[lens == lens.max(), ['stars', 'text']]
# df.loc[lens == lens.min(), ['stars', 'text']]
# lens.mean()
# lens.var()


#train = df.loc[sample, ['stars', 'text']]
#test = df.loc[test_sample, ['stars', 'text']]
train = df.loc[:, ['stars', 'text']]
#test = test.loc[:, ['stars', 'text']]

label_cols = ['1', '2', '3', '4', '5']

for i in label_cols:
    train[i] = (train['stars'] == int(i))
    # test[i] = (test['stars'] == int(i))

re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')


def tokenize(s): return re_tok.sub(r' \1 ', s).split()

# n = df.shape[0]
n = train.shape[0]
# parameters are untuned!
# term frequency–inverse document frequency
vec = TfidfVectorizer(ngram_range=(1,2), tokenizer=tokenize,
                      min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,
                      smooth_idf=1, sublinear_tf=1)
# This creates a sparse matrix with only a small number of non-zero elements
trn_term_doc = vec.fit_transform(train['text'])

import pickle
pickle.dump(vec, open('./NBSVM_model/vectorizer.pk', 'wb'))
#test_term_doc = vec.transform(test['text'])


# the basic naive bayes feature equation
def pr(y_i, y):
    p = x[y == y_i].sum(0)
    return (p+1) / ((y==y_i).sum()+1)

x = trn_term_doc
#test_x = test_term_doc


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



################### prediction part ############################
import pickle
import pandas as pd
# from rake_nltk import Rake
# import nltk
import string, re
# import collections
# import vaderSentiment
# import nltk.sentiment
# from nltk.corpus import sentiwordnet as swn
# from sklearn import linear_model, metrics, datasets
import numpy as np
# import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import time

start = time.time()

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


df = pd.read_csv("test_new.csv")

#train = pd.read_csv("train_new.csv")
#test = pd.read_csv("test_new.csv")

#random.seed(8102)
# select train and test samples
#sample_size = 346378
#test_size = 100000
#
#sample = random.sample(range(df.shape[0]), sample_size)
#
#test_sample = sample[99999:]
#sample = sample[0:99999]

# data insight
# lens = [len(x.split()) for x in df.text]
# lens = pd.Series(lens)
# df.loc[lens == lens.max(), ['stars', 'text']]
# df.loc[lens == lens.min(), ['stars', 'text']]
# lens.mean()
# lens.var()


#train = df.loc[sample, ['stars', 'text']]
#test = df.loc[test_sample, ['stars', 'text']]
test = df.loc[:, [ 'text']]
#test = test.loc[:, ['stars', 'text']]

label_cols = ['1', '2', '3', '4', '5']

#for i in label_cols:
#    test[i] = (test['stars'] == int(i))
    # test[i] = (test['stars'] == int(i))

re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')


#def tokenize(s): return re_tok.sub(r' \1 ', s).split()

# n = df.shape[0]
n = test.shape[0]
# parameters are untuned!
# term frequency–inverse document frequency
vec_name = './NBSVM_model/vectorizer.pk'
vec = pickle.load(open(vec_name, 'rb'))

test_term_doc = vec.transform(test['text'])

preds = np.zeros((test.shape[0], len(label_cols)))

import pickle

for i, j in enumerate(label_cols):
    print('fit', j)
    m_name = './NBSVM_model/m' + j + '.sav'
    r_name = './NBSVM_model/r' + j + '.npy'
    m = pickle.load(open(m_name, 'rb'))
    r = np.asmatrix(np.load(r_name))
    preds[:,i] = m.predict_proba(test_term_doc.multiply(r))[:,1]
    
    

# predictions are the probabilities of stars (1-5)
preds = np.asmatrix(preds)
stars = np.matrix([1, 2, 3, 4, 5]).transpose()

# multiply normalized probabilities with stars (Estimation)
tmp_preds = preds.sum(1)
tmp_preds = preds / tmp_preds
tmp = tmp_preds * stars
#print("est.stars", mse_calculation(test['stars'], tmp))
#
## we multiply each probability with stars
#tmp = preds * stars
## check if prediction follows correct shape
#tmp.shape
#print("prob*stars", mse_calculation(test['stars'], tmp))
#
## get stars with max probability
#tmp = [x.argmax()+1 for x in preds]
#print("max_prob: ", mse_calculation(test['stars'], tmp))
#
#end = time.time()
#print(end - start)