# some attempts to improve the random forest method originally from the word2vec.py work form RQZ.
# this time I am going to

import re
import random
import pandas as pd
# import nltk
from nltk.corpus import stopwords
import nltk.sentiment
import nltk.sentiment.sentiment_analyzer as sentiment
from nltk.stem.porter import PorterStemmer
stem = PorterStemmer()

df = pd.read_csv("/Users/yilixia/Downloads/lyi_small.csv")

random.seed(8102)

sample_size = 50000
sample = random.sample(range(df.shape[0]), sample_size)

test_size = 10000
test_sample = random.sample(range(df.shape[0]), test_size)

# TODO: random forest using lexicons in CountVectorizer
emotion_words = pd.read_csv("Code/Lixia Yi/vader_lexicon.txt", sep="\t", header=None)
emotion_words1 = pd.read_csv("Code/Lixia Yi/SentiNetLexicon.txt", sep=" ", header=None)

emotion_words_list = emotion_words[0].tolist()
emotion_words_list1 = emotion_words1[0].tolist()


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
