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



df = pd.read_csv("/Users/yilixia/Downloads/lyi_small.csv")


random.seed(8102)

sample_size = 50000
sample = random.sample(range(df.shape[0]), sample_size)

test_size = 10000
test_sample = random.sample(range(df.shape[0]), test_size)

# data insight
lens = [len(x.split()) for x in df.text]
lens = pd.Series(lens)
df.loc[lens == lens.max(), ['stars', 'text']]
df.loc[lens == lens.min(), ['stars', 'text']]
lens.mean()
lens.var()


train = df.loc[sample, ['stars', 'text']]
test = df.loc[test_sample, ['stars', 'text']]

label_cols = ['1', '2', '3', '4', '5']

for i in label_cols:
    train[i] = (train['stars'] == int(i))
    # test[i] = (test['stars'] == int(i))



import re, string
re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')


def tokenize(s): return re_tok.sub(r' \1 ', s).split()

n = df.shape[0]
# parameters are untuned!
vec = TfidfVectorizer(ngram_range=(1,2), tokenizer=tokenize,
                      min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,
                      smooth_idf=1, sublinear_tf=1)
# This creates a sparse matrix with only a small number of non-zero elements
trn_term_doc = vec.fit_transform(train['text'])
test_term_doc = vec.transform(test['text'])


# the basic naive bayes feature equation
def pr(y_i, y):
    p = x[y == y_i].sum(0)
    return (p+1) / ((y==y_i).sum()+1)

x = trn_term_doc
test_x = test_term_doc


def get_mdl(y):
    y = y.values
    r = np.log(pr(1,y) / pr(0,y))
    m = LogisticRegression(C=4, dual=True)
    x_nb = x.multiply(r)
    return m.fit(x_nb, y), r



preds = np.zeros((test_size, len(label_cols)))

for i, j in enumerate(label_cols):
    print('fit', j)
    m,r = get_mdl(train[j])
    preds[:,i] = m.predict_proba(test_x.multiply(r))[:,1]


preds = np.asmatrix(preds)
stars = np.matrix([1, 2, 3, 4, 5]).transpose()

tmp = preds * stars
tmp.shape

mse_calculation(train['stars'], tmp)
