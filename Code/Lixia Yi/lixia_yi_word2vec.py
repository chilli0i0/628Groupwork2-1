# some attempts to improve the original word2vec.py work form RQZ.

import re
import random
import pandas as pd
# import nltk
from nltk.corpus import stopwords
import nltk.sentiment
import nltk.sentiment.sentiment_analyzer as sentiment

df = pd.read_csv("/Users/yilixia/Downloads/lyi_small.csv")

random.seed(8102)

sample_size = 10000
sample = random.sample(range(df.shape[0]), sample_size)

sentim_Analyzer = nltk.sentiment.SentimentIntensityAnalyzer()
sentim_Analyzer.polarity_scores(df.loc[1,'text'])

###########################################METHOD1###############################
def review_to_words(raw_review, lexicon):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and
    # the output is a single string (a preprocessed movie review)
    # 没写循环
    # 2. Remove non-letters
    letters_only = re.sub("[^a-zA-Z]", " ", raw_review)
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))
    #
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]
    #
    # 6. Join the words back into one string separated by space,
    # and return the result.
    return (" ".join(meaningful_words))


# Initialize an empty list to hold the clean reviews
clean_train_reviews = []

# Loop over each review; create an index i that goes from 0 to the length
# of the movie review list
train_num = 50000
for i in range(train_num):
    # Call our function for each one, and add the result to the list of
    # clean reviews
    if ((i + 1) % 1000 == 0):
        print("Review %d of %d\n" % (i + 1, train_num))
    clean_train_reviews.append(review_to_words(df.iloc[i, 2]))
##to bag of words
print("Creating the bag of words...\n")
from sklearn.feature_extraction.text import CountVectorizer

# Initialize the "CountVectorizer" object, which is scikit-learn's
# bag of words tool.
vectorizer = CountVectorizer(analyzer="word", \
                             tokenizer=None, \
                             preprocessor=None, \
                             stop_words=None, \
                             max_features=7000)
# 怎么多加停词

train_data_features = vectorizer.fit_transform(clean_train_reviews)

# Numpy arrays are easy to work with, so convert the result to an
# array
train_data_features = train_data_features.toarray()

print(train_data_features.shape)
vocab = vectorizer.get_feature_names()  # 按字母排列顺序的
import numpy as np

# Sum up the counts of each vocabulary word
dist = np.sum(train_data_features, axis=0)

# For each, print the vocabulary word and the number of times it
# appears in the training set
for tag, count in zip(vocab, dist):
    print(count, tag)

from sklearn.ensemble import RandomForestClassifier

# Initialize a Random Forest classifier with 100 trees
forest = RandomForestClassifier(n_estimators=100)

# Fit the forest to the training set, using the bag of words as
# features and the sentiment labels as the response variable
#
# This may take a few minutes to run
forest = forest.fit(train_data_features, df.iloc[0:train_num, 0])

test = df.iloc[train_num + 1:train_num + 501, 2]

# Verify that there are 25,000 rows and 2 columns
print(test.shape)

# Create an empty list and append the clean reviews one by one
num_reviews = len(test)
clean_test_reviews = []

print("Cleaning and parsing the test set movie reviews...\n")
for i in range(num_reviews):
    if ((i + 1) % 1000 == 0):
        print("Review %d of %d\n" % (i + 1, num_reviews))
    clean_review = review_to_words(test.iloc[i])
    clean_test_reviews.append(clean_review)

# Get a bag of words for the test set, and convert to a numpy array
test_data_features = vectorizer.transform(clean_test_reviews)
test_data_features = test_data_features.toarray()

# Use the random forest to make sentiment label predictions
result = forest.predict(test_data_features)

true_test = df.iloc[train_num + 1:train_num + 501, 0]
fake_sum = 0
for i in range(500):
    if result[i] == true_test.iloc[i]:
        fake_sum += 1
accuracy = fake_sum / 500
accuracy