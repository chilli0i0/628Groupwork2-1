# some attempts to improve the random forest method originally from the word2vec.py work form RQZ.
# this time I am going to

import re
import random
import pandas as pd
# import nltk
from nltk.corpus import stopwords
import nltk.sentiment
import nltk.sentiment.sentiment_analyzer as sentiment
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import math
from sklearn.feature_extraction.text import CountVectorizer
import string
from rake_nltk import Rake

# stemmer
stem = PorterStemmer()

df = pd.read_csv("/Users/yilixia/Downloads/lyi_small.csv")


def mse_calculation(target,prediction):
    prediction_restrain = list()
    for i in prediction:
        if i >= 1 and i <= 5:
            prediction_restrain.append(i)
        elif i < 1:
            prediction_restrain.append(1)
        else:
            prediction_restrain.append(5)
    return math.sqrt(sum(map(lambda x, y: np.square(x-y), target, prediction_restrain))/len(target))




random.seed(8102)

sample_size = 50000
sample = random.sample(range(df.shape[0]), sample_size)

test_size = 10000
test_sample = random.sample(range(df.shape[0]), test_size)


# TODO: random forest using lexicons in CountVectorizer
emotion_words = pd.read_csv("Code/Lixia Yi/vader_lexicon.txt", sep="\t", header=None)
emotion_words1 = pd.read_csv("Code/Lixia Yi/SentiNetLexicon.txt", sep=" ", header=None)

emotion_words_list = emotion_words[0].tolist() + emotion_words1[0].tolist()

# extract key words from sentences
stop_words = ['a', 'able', 'about', 'across', 'after', 'all', 'almost', 'also', 'am', 'among', 'an', 'and', 'any',
              'are', 'as', 'at', 'be', 'because', 'been', 'by', 'did', 'else', 'ever', 'every', 'for', 'from',
              'get', 'got', 'had', 'has', 'have', 'how', 'however', 'i', 'if',
              'in', 'into', 'is', 'it', 'its', 'just', 'least', 'let', 'may', 'me', 'might', 'my', 'of', 'off',
              'on', 'or', 'other', 'our', 'own', 'rather', 'said', 'say', 'says', 'she', 'should', 'since', 'so',
              'than', 'that', 'the', 'their', 'them', 'then', 'there', 'these', 'they', 'this', 'tis', 'to', 'was',
              'us', 'was', 'we', 'were', 'what', 'when', 'where', 'while', 'who', 'whom', 'why', 'will', 'would',
              'yet', 'you', 'your', 'they', 'look', 'good', 'a', 'able', 'about', 'across', 'after', 'all',
              'almost', 'also', 'am', 'among', 'an', 'and', 'any', 'are', 'as', 'at', 'be', 'because', 'been', 'by',
              'did', 'else', 'ever', 'every', 'for', 'from', 'get', 'got', 'had', 'has', 'have',
              'how', 'however', 'i', 'if', 'in', 'into', 'is', 'it', 'its', 'just', 'least',
              'let', 'may', 'me', 'might', 'my', 'of', 'off', 'on', 'or', 'other', 'our', 'own', 'rather', 'said',
              'say', 'says', 'she', 'should', 'since', 'so', 'than', 'that', 'the', 'their', 'them', 'then',
              'there', 'these', 'they', 'this', 'tis', 'to', 'was', 'us', 'was', 'we', 'were', 'what', 'when',
              'where', 'while', 'who', 'whom', 'why', 'will', 'would', 'yet', 'you', 'your', '!', '@', '#', '"',
              '$', '(', '.', ')']

r = Rake(stop_words, string.punctuation)
all_keywords = list()
all_keywords_split = list()
for i in range(sample_size):
    words = [i.lower() for i in word_tokenize(df["text"][i])]
    sentence_re_stopwords = " ".join([token for token in words if token not in set(string.punctuation)])
    r.extract_keywords_from_text(sentence_re_stopwords)
    phrases = r.get_ranked_phrases()
    scores = r.get_ranked_phrases_with_scores()
    all_keywords.append([scores[i] for i in range(len(phrases)) if scores[i][0] > 1])
    all_keywords_split.append(
        [(scores[i][0], nltk.word_tokenize(phrases[i])) for i in range(len(phrases)) if scores[i][0] > 1])

import collections

all_keywords_each = list()
for i in all_keywords_split:
    for j in i:
        for l in range(len(j[1])):
            if j[1][l].isalpha():
                all_keywords_each.append(j[1][l])

keywords_in_each_review = list()
for i in all_keywords_split:
    tmp = list()
    score_list = [i[k][0] for k in range(len(i))]
    for index, j in enumerate(i):
        for l in range(len(j[1])):
            if j[1][l].isalpha():
                tmp.append((score_list[index], j[1][l]))
    keywords_in_each_review.append(tmp)

keywords_in_each_review_small = collections.Counter(all_keywords_each).most_common()[0:19890]

# extract emotion words from keywords
emotion_words_times = [all_keywords_each.count(i) for i in emotion_words_list]
emotion_words_selected = [emotion_words_list[i] for i in range(len(emotion_words_list)) if emotion_words_times[i] > 0]


# Select emotion words from
def select_from_lexicon(raw_review, lexicon):
    return None





print("Creating the bag of words...\n")

# Initialize the "CountVectorizer" object, which is scikit-learn's
# bag of words tool.
vectorizer = CountVectorizer(analyzer="word",
                             tokenizer=None,
                             preprocessor=None,
                             stop_words=None,
                             max_features=5000,
                             max_df=0.08,
                             input=keywords_in_each_review_small,
                             lowercase=True)


train_data_features = vectorizer.fit_transform(df.loc[sample, 'text'])

# Numpy arrays are easy to work with, so convert the result to an
# array
train_data_features = train_data_features.toarray()

print(train_data_features.shape)
vocab = vectorizer.get_feature_names()  # 按字母排列顺序的

# Sum up the counts of each vocabulary word
dist = np.sum(train_data_features, axis=0)

# For each, print the vocabulary word and the number of times it
# appears in the training set
for tag, count in zip(vocab, dist):
    print(count, tag)

print("Fit a RandomForest")
# Initialize a Random Forest classifier with 100 trees
forest = RandomForestClassifier(n_estimators=100)

# Fit the forest to the training set, using the bag of words as
# features and the sentiment labels as the response variable
#
# This may take a few minutes to run
forest = forest.fit(train_data_features, df.loc[sample, 'stars'])

# get the test text
test = df.loc[test_sample, 'text']

# Verify its shape
print(test.shape)

# Create an empty list and append the clean reviews one by one
# clean_test_reviews = []
#
# print("Cleaning and parsing the test set reviews...\n")
# for i in range(test_size):
#     if ((i + 1) % 1000 == 0):
#         print("Review %d of %d\n" % (i + 1, test_size))
#     clean_review = review_to_words(test.loc[test_sample[i]], stem)
#     clean_test_reviews.append(clean_review)

# Get a bag of words for the test set, and convert to a numpy array
test_data_features = vectorizer.transform(df.loc[test_sample, 'text'])
test_data_features = test_data_features.toarray()

# Sum up the counts of each vocabulary word
dist = np.sum(test_data_features, axis=0)
for tag, count in zip(vocab, dist):
    print(count, tag)

# Use the random forest to make sentiment label predictions
result = forest.predict(test_data_features)

true_test = df.loc[test_sample, 'stars']

fake_sum = 0
for i in range(500):
    if result[i] == true_test.iloc[i]:
        fake_sum += 1
accuracy = fake_sum / 500
accuracy

mse_calculation(result, true_test)
