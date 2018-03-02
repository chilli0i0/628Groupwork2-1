# read data
import pandas as pd
# from rake_nltk import Rake
# import nltk
# import string
# import collections
# import vaderSentiment
import random
import nltk.sentiment
# from nltk.corpus import sentiwordnet as swn
# from sklearn import linear_model, metrics, datasets
import numpy as np
# import matplotlib.pyplot as plt


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

sentim_Analyzer = nltk.sentiment.SentimentIntensityAnalyzer()

random.seed(8102)

sample_size = 50000
sample = random.sample(range(df.shape[0]), sample_size)

test_size = 10000
test_sample = random.sample(range(df.shape[0]), test_size)

pos = []
neg = []
neu = []
compound = []
length = []
for i in range(sample_size):
    if ((i + 1) % 1000 == 0):
        print("Review %d of %d\n" % (i + 1, sample_size))
    sentim_tmp = sentim_Analyzer.polarity_scores(df.loc[sample[i], 'text'])
    pos.append(sentim_tmp.get('pos'))
    neg.append(sentim_tmp.get('neg'))
    neu.append(sentim_tmp.get('neu'))
    compound.append(sentim_tmp.get('compound'))
    length.append(len(df.loc[sample[i], 'text']))


sample_stars = (df.loc[sample, 'stars'])

train = np.column_stack((pos, neg, neu, compound, length))

# linear regression
from sklearn.linear_model import LinearRegression
lm = LinearRegression(fit_intercept=True, normalize=True, copy_X=True, n_jobs=1)
lm_fit = lm.fit(train, sample_stars)


# test model
pos = []
neg = []
neu = []
compound = []
length = []
for i in range(test_size):
    if ((i + 1) % 1000 == 0):
        print("Review %d of %d\n" % (i + 1, test_size))
    sentim_tmp = sentim_Analyzer.polarity_scores(df.loc[test_sample[i], 'text'])
    pos.append(sentim_tmp.get('pos'))
    neg.append(sentim_tmp.get('neg'))
    neu.append(sentim_tmp.get('neu'))
    compound.append(sentim_tmp.get('compound'))
    length.append(len(df.loc[test_sample[i], 'text']))

test = np.column_stack((pos, neg, neu, compound, length))

test_stars = (df.loc[test_sample, 'stars'])

lm_predict = lm.predict(test)
mse_calculation([float(i) for i in test_stars], list(lm_predict))

# MSE:
# 0.9885342685005917 with pos, neg, neu and compound alone
# 0.9738422761451919 after adding length as factor
