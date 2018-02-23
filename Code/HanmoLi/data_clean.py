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
stopset = set(nltk.corpus.stopwords.words('english'))
stopset.add("food")
stopset.add("would")
stopset.add("go")
stopset.add("get")
stopset.add("also")
stopset.add("got")
stopset.add("every")
stopset.add("...")
stopset.add("!!!")
for i in string.punctuation:
    stopset.add(i)
# use nlty to remove stopwords first
    
words = [i.lower() for i in word_tokenize(df["text"][1])]
r = Rake(stopset,string.punctuation)
sentence_re_stopwords = " ".join([token for token in words if token not in set(string.punctuation)])
r.extract_keywords_from_text(df["text"][1])
phrases = r.get_ranked_phrases()
scores = r.get_ranked_phrases_with_scores()



# extract key words from sentences

r = Rake(stopset,string.punctuation)
all_keywords = list()
all_keywords_split = list()
for i in range(100):
    r.extract_keywords_from_text(df["text"][i])
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
        return sum(map(lambda i:(i-min_score)/(max_score-min_score),score_unst))/len(score_unst)
    else:
        return None

calculate_relative_score(keywords_in_each_review,6,emotion_words_selected[78])

def calculate_identital_score(emotion_word,emotion_related_scores):
    # str emotion_word
    # dict[str:int] emotion_related_scores 
    return emotion_related_scores[emotion_word]/4

calculate_identital_score(emotion_words_selected[7],emotion_related_scores)
    
    

