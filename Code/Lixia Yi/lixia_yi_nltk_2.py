# read data
import pandas as pd
from rake_nltk import Rake
import nltk
import string
import collections
import vaderSentiment
import random
import nltk.sentiment
from nltk.corpus import sentiwordnet as swn
from sklearn import linear_model, metrics, datasets
import numpy as np
import matplotlib.pyplot as plt



df = pd.read_csv("/Users/yilixia/Downloads/lyi_small.csv")

sentim_Analyzer = nltk.sentiment.SentimentIntensityAnalyzer()

random.seed(8102)

sample_size = 10000
sample = random.sample(range(df.shape[0]), sample_size)

pos = []
neg = []
neu = []
compound = []
for i in range(sample_size):
    sentim_tmp = sentim_Analyzer.polarity_scores(df.loc[sample[i], 'text'])
    pos.append(sentim_tmp.get('pos'))
    neg.append(sentim_tmp.get('neg'))
    neu.append(sentim_tmp.get('neu'))
    compound.append(sentim_tmp.get('compound'))


sample_stars = list(df.loc[sample, 'stars'])

