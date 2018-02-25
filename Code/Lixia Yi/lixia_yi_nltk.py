# TODO: Sample from the dataset, analyze it through nltk, try to implement a regression model on it
# TODO: and test it on a sampled testset
#

# read data
import pandas as pd
from rake_nltk import Rake
import nltk
import string
import collections
import vaderSentiment
import random
import nltk.sentiment
from sklearn import linear_model, metrics, datasets
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("/Users/yilixia/Downloads/train_data.csv")
cities = ["Edinburgh", "Karlsruhe", "Montreal", "Waterloo", "Pittsburgh", "Charlotte", "Urbana-Champaign", "Phoenix",
          "Las Vegas", "Madison", "Cleveland"]
colnames = df.columns.values.tolist()
new_data = pd.DataFrame(columns=[colnames])
for i in cities:
    new_data = pd.concat([new_data, df[df["city"] == i]], axis=1)

# df["stars"][df["text"].str.contains("decilious")].count()
# df["stars"][df["text"].str.contains("minutes")].mean()
# df["stars"][df["text"].str.contains("minutes")].hist()
# df["text"][df["stars"]==1].str.len().mean()
# df["text"].to_csv(r'C:\Users\lihan\OneDrive - UW-Madison\window\STAT 628\Module_2\reviews.txt')


# reviews clean and key words selection
# stopset = set(nltk.corpus.stopwords.words('english'))
# stopset.add("us")
# stopset.add("would")
# stopset.add("go")
# stopset.add("get")
# stopset.add("also")
# stopset.add("got")
# stopset.add("every")

#
# r = Rake(stopset, string.punctuation + "?")
# all_keywords = list()
# all_keywords_split = list()
# for i in range(100):
#     r.extract_keywords_from_text(df["text"][i])
#     phrases = r.get_ranked_phrases()
#     scores = r.get_ranked_phrases_with_scores()
#     all_keywords.append([phrases[i] for i in range(len(phrases)) if scores[i][0] > 1])
#     all_keywords_split.append([nltk.word_tokenize(phrases[i]) for i in range(len(phrases)) if scores[i][0] > 1])
#
#
# all_keywords_each = list()
# for i in all_keywords_split:
#     for j in i:
#         for l in range(len(j)):
#             all_keywords_each.append(j[l])
#
# collections.Counter(all_keywords_each).most_common()[0:99]
#
# emotion_words = pd.read_csv("Code/HanmoLi/vader_lexicon.txt", sep="\t", header=None)
#
# emotion_words_list = emotion_words[0].tolist()
# emotion_scores_list = emotion_words[1].tolist()
#
# emotion_words_times = [all_keywords_each.count(i) for i in emotion_words_list]
# emotion_words_selected = [emotion_words_list[i] for i in range(len(emotion_words_list)) if emotion_words_times[i] > 0]


# My code

sentim_Analyzer = nltk.sentiment.SentimentIntensityAnalyzer()

#####
# sample code
lines_list = nltk.tokenize.sent_tokenize(df.loc[1, 'text'])

for sentence in lines_list:
    print(sentence)
    ss = sentim_Analyzer.polarity_scores(sentence)
    for k in sorted(ss):
        print('{0}: {1}, '.format(k, ss[k]), end='')
        print()
#####
#
random.seed(8102)

sample_size = 10000
sample = random.sample(range(new_data.shape[0]), sample_size)

compound = []
for i in range(sample_size):
    lines_list = nltk.tokenize.sent_tokenize(df.loc[sample[i], 'text'])
    comp_tmp = []
    for sentence in lines_list:
        sentiment_tmp = sentim_Analyzer.polarity_scores(sentence)
        comp_tmp.append(sentiment_tmp.get('compound'))
    if sum([abs(x) for x in comp_tmp]) == 0:
        compound.append(sum(comp_tmp))
    else:
        compound.append(sum(comp_tmp)/sum([abs(x) for x in comp_tmp]))

sample_stars = np.asarray(df.loc[sample, 'stars'])
compound = np.asarray(compound)
# do compound and stars have a (linear) relationship?
# plot box-plot:
data = []
for i in range(1, 6):
    data.append(compound[sample_stars == i])

plt.boxplot(data)
plt.title("Sentiment Scores against Stars")
plt.ylabel("Sentiment")
plt.xlabel("Stars")
plt.show()

# regression
# errors appear, need to fix!!
regr = linear_model.LinearRegression()

regr.fit(compound, sample_stars)  # TODO: cannot fit it, fix the problem!

# test on new sample set
new_sample = random.sample(range(new_data.shape[0]), 100)

compound = []
for i in range(100):
    lines_list = nltk.tokenize.sent_tokenize(df.loc[new_sample[i], 'text'])
    comp_tmp = []
    for sentence in lines_list:
        sentiment_tmp = sentim_Analyzer.polarity_scores(sentence)
        comp_tmp.append(sentiment_tmp.get('compound'))
    compound.append(sum(comp_tmp))

sample_stars = np.asarray(df.loc[new_sample, 'stars'])
compound = np.asarray(compound)

stars_pred = regr.predict(compound.reshape(-1, 1))