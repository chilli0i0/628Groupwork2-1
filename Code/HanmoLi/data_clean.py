# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 20:09:03 2018

@author: lihan
"""



# read data
import pandas as pd
df=pd.read_csv("train_data.csv")
cities=["Edinburgh","Karlsruhe","Montreal","Waterloo","Pittsburgh","Charlotte","Urbana-Champaign","Phoenix","Las Vegas","Madison","Cleveland"]
colnames=df.columns.values.tolist()
new_data=pd.DataFrame(columns=[colnames])
for i in cities:
    new_data=pd.concat([new_data,df[df["city"]==i]],axis=1)

#df["stars"][df["text"].str.contains("decilious")].count()
#df["stars"][df["text"].str.contains("minutes")].mean()
#df["stars"][df["text"].str.contains("minutes")].hist()
#df["text"][df["stars"]==1].str.len().mean()
#df["text"].to_csv(r'C:\Users\lihan\OneDrive - UW-Madison\window\STAT 628\Module_2\reviews.txt')


# reviews clean and key words selection

from rake_nltk import Rake
import nltk
import string
from nltk.tokenize import sent_tokenize, word_tokenize
import re
stopset = set(nltk.corpus.stopwords.words('english'))
#stopset.add("food")
#stopset.add("would")
#stopset.add("go")
#stopset.add("get")
#stopset.add("also")
#stopset.add("got")
#stopset.add("every")
#stopset.add("...")
#stopset.add("!!!")
#for i in string.punctuation:
#    stopset.add(i)
# use nlty to remove stopwords first
    
words = [i.lower() for i in word_tokenize(df["text"][12])]
r = Rake(stopset,string.punctuation)
sentence_re_stopwords = " ".join([token for token in words if token not in set(string.punctuation)])
r.extract_keywords_from_text(sentence_re_stopwords)
phrases = r.get_ranked_phrases()
scores = r.get_ranked_phrases_with_scores()

# extract stars in the reviews:
stars = list()
pattern1 = re.compile('(\d+\.\d*) star')
pattern2 = re.compile('(\d*) star')
for i in range(1000):
    words = [i.lower() for i in word_tokenize(df["text"][i])]
    sentence_re_stopwords = " ".join([token for token in words if token not in set(string.punctuation)])
    sen_re1 = pattern1.search(sentence_re_stopwords)
    sen_re2 = pattern2.search(sentence_re_stopwords) 
    if sen_re1 is None:
        stars.append(0)
        continue
    else:
        stars.append(float(sen_re1.group(1)))
        continue
    if sen_re2 is None:
        stars.append(0)
        continue
    else:
        stars.append(float(sen_re2.group(1)))

[i for i in stars if i != 0]

# extract key words from sentences

r = Rake(stopset,string.punctuation)
all_keywords = list()
all_keywords_split = list()
for i in range(1000):
    words = [i.lower() for i in word_tokenize(df["text"][i])]
    sentence_re_stopwords = " ".join([token for token in words if token not in set(string.punctuation)])
    r.extract_keywords_from_text(sentence_re_stopwords)
    phrases = r.get_ranked_phrases()
    scores = r.get_ranked_phrases_with_scores()
    all_keywords.append([scores[i] for i in range(len(phrases)) if scores[i][0]>1])
    all_keywords_split.append([(scores[i][0],nltk.word_tokenize(phrases[i])) for i in range(len(phrases)) if scores[i][0]>1])

import collections

all_keywords_each = list()
for  i in all_keywords_split:
    for j in i:
        for l in range(len(j[1])):
            if j[1][l].isalpha():
                all_keywords_each.append(j[1][l])
            
keywords_in_each_review = list()
for  i in all_keywords_split:
    tmp = list()
    score_list = [i[k][0] for k in range(len(i))]
    for index,j in enumerate(i):
        for l in range(len(j[1])):
            if j[1][l].isalpha():
                tmp.append((score_list[index],j[1][l]))
    keywords_in_each_review.append(tmp)
            
collections.Counter(all_keywords_each).most_common()[0:99]



# extract emotion words from keywords

emotion_words = pd.read_csv("vader_lexicon.txt",sep = "\t",header=None)

emotion_words_list = emotion_words[0].tolist()
emotion_scores_list = emotion_words[1].tolist()

emotion_words_times = [all_keywords_each.count(i) for i in emotion_words_list]
emotion_words_selected = [emotion_words_list[i] for i in range(len(emotion_words_list)) if emotion_words_times[i]>0]
emotion_related_scores = {emotion_words_list[i]:emotion_scores_list[i] for i in range(len(emotion_words_list)) if emotion_words_times[i]>0}


emotion_words_in_reviews = list()
for index,i in enumerate(emotion_words_selected):
    tmp = list()
    for j in range(len(keywords_in_each_review)):
        tmp.append([i[1] for i in keywords_in_each_review[j]].count(i))
    emotion_words_in_reviews.append(("critical_words"+str(index),tmp))

emotion_words_in_reviews_df = pd.DataFrame.from_items(emotion_words_in_reviews)

emotion_words_in_reviews_df.head()

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

calculate_relative_score(keywords_in_each_review,6,emotion_words_selected[78])

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

stars = list(df['stars'][0:1000])

import matplotlib.pyplot as plt
plt.boxplot([[pone_diff[i] for i in range(len(stars)) if stars[i] == j] for j in range(1,6)])
plt.hist([stars[i] for i in range(len(stars)) if pone_index[i]==0])
plt.hist([stars[i] for i in range(len(stars)) if pone_index[i]==1])
plt.plot(pone_diff,stars,'.')

# debug
[i for i in range(len(stars)) if pone_index[i]==0]
[stars[i] for i in range(len(stars)) if pone_index[i]==0]
test_index = 73
df['text'][test_index]
df['stars'][test_index]
keywords_in_each_review[test_index]
pone_diff[test_index]

accurate_rate = len([i for i in range(len(stars)) if pone_index[i]==1 and stars[i]>=3])/len(pone_index)

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
def mse_calculation(target,prediction):
    return math.sqrt(sum(map(lambda x,y:(x-y)**2,target,prediction))/len(target))


# fit a simple linear regression model

import statsmodels.api as sm
import numpy as np
import pandas as pd

variables_dict = {"positive":posi_score,"negative":negative_score,'difference':pone_diff,"sign":pone_index}
variables = pd.DataFrame.from_dict(variables_dict)
target = pd.DataFrame(stars,columns=['stars'])
model = sm.OLS(target, variables).fit()
predictions = model.predict(variables) # make the predictions by the model
plt.plot(list(predictions),stars,'.')
# Print out the statistics
model.summary()
target['prediction'] = predictions
mse_calculation([float(i) for i in target['stars']],list(target['prediction']))
