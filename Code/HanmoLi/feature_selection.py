#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 22:18:08 2018

@author: hanmoli

pls run data_clean before execute this script
"""

# calculate four features: 1. Score for positive word 2. Score for negative word 
# 3. whether it's positive 4. degree of positive

def calculate_relative_score(keywords_in_each_review,ith_review,emotion_word):
    # list[list] keywords_in_each_review
    # int ith_review
    # str jth_emotion_word
    keyword_scores_in_ireview = keywords_in_each_review[ith_review]
    keywords = [i[1] for i in keyword_scores_in_ireview]
    if keywords.count(emotion_word) != 0:
        scores = [i[0] for i in keyword_scores_in_ireview]
        max_score = max(scores)
        min_score = min(scores)
        score_unst = [scores[i] for i in range(len(keywords)) if keywords[i]==emotion_word]
        return sum(map(lambda i:(i-min_score+1)/(max_score-min_score+1),score_unst))/len(score_unst)
    else:
        return None

calculate_relative_score(keywords_in_each_review,1,emotion_words_selected[78])

def calculate_identital_score(emotion_word,emotion_related_scores):
    # str emotion_word
    # dict[str:int] emotion_related_scores 
    return emotion_related_scores[emotion_word]/4

calculate_identital_score(emotion_words_selected[7],emotion_related_scores)


# select positive words:
positive_words_selected = list()
negative_words_selected = list()
for i in emotion_words_selected:
    if calculate_identital_score(i,emotion_related_scores)>0:
        positive_words_selected.append(i)
    else:
        negative_words_selected.append(i)


def calculate_sent_posi_score(keywords_in_each_review,ith_sentence, positive_words_selected,emotion_related_scores):
    relative_score = [0]
    identical_score = [0]
    sentence = keywords_in_each_review[ith_sentence]
    for i in positive_words_selected:
        if i in [j[1] for j in sentence]:
            relative_score.append(calculate_relative_score(keywords_in_each_review,ith_sentence, i)) 
            identical_score.append(calculate_identital_score(i,emotion_related_scores))
    return sum(map(lambda x,y: x*y, relative_score,identical_score))/len(relative_score)
    
def calculate_sent_negative_score(keywords_in_each_review,ith_sentence, negative_words_selected,emotion_related_scores):
    relative_score = [0]
    identical_score = [0]
    sentence = keywords_in_each_review[ith_sentence]
    for i in negative_words_selected:
        if i in [j[1] for j in sentence]:
            relative_score.append(calculate_relative_score(keywords_in_each_review,ith_sentence, i)) 
            identical_score.append(calculate_identital_score(i,emotion_related_scores))
    return sum(map(lambda x,y: abs(x*y), relative_score,identical_score))/len(relative_score)
 
calculate_sent_posi_score(keywords_in_each_review,2, positive_words_selected,emotion_related_scores)

calculate_sent_negative_score(keywords_in_each_review,2, negative_words_selected,emotion_related_scores)


# degree of positive for all sentences
posi_score = list()
for i in range(len(keywords_in_each_review)):
    posi_score.append(calculate_sent_posi_score(keywords_in_each_review,i, positive_words_selected,emotion_related_scores))


# degree of negative for all sentences
negative_score = list()
for i in range(len(keywords_in_each_review)):
    negative_score.append(calculate_sent_negative_score(keywords_in_each_review,i, negative_words_selected,emotion_related_scores))

# whether it is positive(1) or negative(0)
pone_index = list()
for i in range(len(keywords_in_each_review)):
    if posi_score[i] >= negative_score[i]:
        pone_index.append(1)
    else:
        pone_index.append(0)

# the difference between positive and negative
pone_diff = list()
for i in range(len(keywords_in_each_review)):
    pone_diff.append(posi_score[i]-negative_score[i])

#stars = list(df['stars'][0:train_num])

#import matplotlib.pyplot as plt
#plt.boxplot([[pone_diff[i] for i in range(len(stars)) if stars[i] == j] for j in range(1,6)])
#plt.hist([stars[i] for i in range(len(stars)) if pone_index[i]==0])
#plt.hist([stars[i] for i in range(len(stars)) if pone_index[i]==1])
#plt.plot(pone_diff,stars,'.')

# debug
#[i for i in range(len(stars)) if pone_index[i]==0]
#[stars[i] for i in range(len(stars)) if pone_index[i]==0]
#test_index = 38
#df['text'][test_index]
#df['stars'][test_index]
#keywords_in_each_review[test_index]
#pone_diff[test_index]
#
#accurate_rate = len([i for i in range(len(stars)) if pone_index[i]==1 and stars[i]>=3])/len(pone_index)

"""
interesting discoveries:
    
19th review: 'never go wrong' is positive but 'never' and 'wrong' are both negative

38th review: long and boring review. but at the end of this review, the author directly gives
the scores

73th review: 'dead quiet' is positive but 'dead' itself is strongly negative. My algorithm gives the two 'great'
at the beginning of the review the lowest relative score.


From the test, the results of Reka for some sentences are not accurate. That is, Reka sometimes give
the key words relatively low weight comparing to other unimportant words.
"""
import math
import numpy as np
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


# fit a simple linear regression model

import statsmodels.api as sm
import numpy as np
import pandas as pd

variables_dict = {"positive":posi_score,"negative":negative_score,'difference':pone_diff,"sign":pone_index}
variables = pd.DataFrame.from_dict(variables_dict)


# calculate some simple features

# number of the words in reviews
import nltk
from nltk.tokenize import sent_tokenize
num_words = [len(nltk.word_tokenize(df.iloc[i,2])) for i in range(df.shape[0])]
num_sentence = [len(sent_tokenize(df.iloc[i,2])) for i in range(df.shape[0])]
simple_dict = {"num_words":num_words,"num_sentence":num_sentence}
simple = pd.DataFrame.from_dict(simple_dict)

from nltk.sentiment import SentimentIntensityAnalyzer
sentim_Analyzer = SentimentIntensityAnalyzer()
pos = []
neg = []
neu = []
compound = []
length = []
sample_size = df.shape[0]
for i in range(sample_size):
    if ((i + 1) % 1000 == 0):
        print("Review %d of %d\n" % (i + 1, sample_size))
    sentim_tmp = sentim_Analyzer.polarity_scores(df.iloc[i, 2])
    pos.append(sentim_tmp.get('pos'))
    neg.append(sentim_tmp.get('neg'))
    neu.append(sentim_tmp.get('neu'))
    compound.append(sentim_tmp.get('compound'))
    length.append(len(df.iloc[i,2].split()))
ylx_dict = {"pos":pos,"neg":neg,"neu":neu,"compound":compound,"length":length}
ylx = pd.DataFrame.from_dict(ylx_dict)






# save variable
variables.to_pickle('./features/emotion_features')
simple.to_pickle('./features/simple_features')
ylx.to_pickle('./features/ylx')
#target = pd.DataFrame(stars,columns=['stars'])
#x = variables
#model = sm.OLS(target, x).fit()
#predictions = model.predict(x) # make the predictions by the model
#plt.plot(list(predictions),stars,'.')
## Print out the statistics
#model.summary()
#target['prediction'] = predictions
#mse_calculation([float(i) for i in target['stars']],list(target['prediction']))
