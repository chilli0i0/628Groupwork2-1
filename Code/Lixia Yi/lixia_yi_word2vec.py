# some attempts to improve the original word2vec.py work form RQZ.

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

# never used yet.
sentim_Analyzer = nltk.sentiment.SentimentIntensityAnalyzer()
sentim_Analyzer.polarity_scores(df.loc[1, 'text'])

###########################################METHOD1###############################
def review_to_words(raw_review, stem):
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
    stop_words = ['a', 'able', 'about', 'across', 'after', 'all', 'almost', 'also', 'am', 'among', 'an', 'and', 'any',
                  'are', 'as', 'at', 'be', 'because', 'been', 'by', 'did', 'else', 'ever', 'every', 'for', 'from',
                  'get', 'got', 'had', 'has', 'have', 'how', 'however', 'i', 'if',
                  'in', 'into', 'is', 'it', 'its', 'just', 'least', 'let', 'may', 'me', 'might', 'my', 'of', 'off',
                  'on', 'or', 'other', 'our', 'own', 'rather', 'said', 'say', 'says', 'she', 'should', 'since', 'so',
                  'than', 'that', 'the', 'their', 'them', 'then', 'there', 'these', 'they', 'this', 'tis', 'to', 'was',
                  'us', 'was', 'we', 'were', 'what', 'when', 'where', 'while', 'who', 'whom', 'why', 'will', 'would',
                  'yet', 'you', 'your', 'They', 'Look', 'Good', 'A', 'Able', 'About', 'Across', 'After', 'All',
                  'Almost', 'Also', 'Am', 'Among', 'An', 'And', 'Any', 'Are', 'As', 'At', 'Be', 'Because', 'Been', 'By',
                  'Did', 'Else', 'Ever', 'Every', 'For', 'From', 'Get', 'Got', 'Had', 'Has', 'Have',
                  'How', 'However', 'I', 'If', 'In', 'Into', 'Is', 'It', 'Its', 'Just', 'Least',
                  'Let', 'May', 'Me', 'Might', 'My', 'Of', 'Off', 'On', 'Or', 'Other', 'Our', 'Own', 'Rather', 'Said',
                  'Say', 'Says', 'She', 'Should', 'Since', 'So', 'Than', 'That', 'The', 'Their', 'Them', 'Then',
                  'There', 'These', 'They', 'This', 'Tis', 'To', 'Was', 'Us', 'Was', 'We', 'Were', 'What', 'When',
                  'Where', 'While', 'Who', 'Whom', 'Why', 'Will', 'Would', 'Yet', 'You', 'Your', '!', '@', '#', '"',
                  '$', '(', '.', ')']

    #
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stop_words]
    #
    # 6. Stem the words.
    meaningful_words = [stem.stem(w) for w in meaningful_words]

    # 7. Join the words back into one string separated by space,
    # and return the result.
    return (" ".join(meaningful_words))


# Initialize an empty list to hold the clean reviews
clean_train_reviews = []

# Loop over each review; create an index i that goes from 0 to the length
# of the movie review list
for i in range(sample_size):
    # Call our function for each one, and add the result to the list of
    # clean reviews
    if ((i + 1) % 1000 == 0):
        print("Review %d of %d\n" % (i + 1, sample_size))
    clean_train_reviews.append(review_to_words(df.loc[sample[i], 'text'], stem))
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
forest = forest.fit(train_data_features, df.loc[sample, 'stars'])

# get the test text
test = df.loc[test_sample, 'text']

# Verify its shape
print(test.shape)

# Create an empty list and append the clean reviews one by one
clean_test_reviews = []

print("Cleaning and parsing the test set reviews...\n")
for i in range(test_size):
    if ((i + 1) % 1000 == 0):
        print("Review %d of %d\n" % (i + 1, test_size))
    clean_review = review_to_words(test.loc[test_sample[i]], stem)
    clean_test_reviews.append(clean_review)

# Get a bag of words for the test set, and convert to a numpy array
test_data_features = vectorizer.transform(clean_test_reviews)
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

import math
def mse_calculation(target,prediction):
    return math.sqrt(sum(map(lambda x,y:(x-y)**2,target,prediction))/len(target))

mse_calculation(result,true_test)

###########################################METHOD2###############################

# Part two for word2vec
# need to fix problems of  other language
import nltk


def review_to_wordlist(review, remove_stopwords=False):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    #
    # 1. Remove HTML
    #
    # 2. Remove non-letters
    review_text = re.sub("[^a-zA-Z]", " ", review)
    #
    # 3. Convert words to lower case and split them
    words = review_text.lower().split()
    #
    # 4. Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    #
    # 5. Return a list of words
    return (words)


tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


def review_to_sentences(review, tokenizer, remove_stopwords=False):
    # Function to split a review into parsed sentences. Returns a
    # list of sentences, where each sentence is a list of words
    #
    # 1. Use the NLTK tokenizer to split the paragraph into sentences
    raw_sentences = tokenizer.tokenize(review.strip())
    #
    # 2. Loop over each sentence
    sentences = []
    for raw_sentence in raw_sentences:
        # If a sentence is empty, skip it
        if len(raw_sentence) > 0:
            # Otherwise, call review_to_wordlist to get a list of words
            sentences.append(review_to_wordlist(raw_sentence, \
                                                remove_stopwords))
    #
    # Return the list of sentences (each sentence is a list of words,
    # so this returns a list of lists
    return sentences


sentences = []  # Initialize an empty list of sentences

print("Parsing sentences from training set")
for i in range(train_num):
    sentences += review_to_sentences(df.iloc[i, 2], tokenizer)
# Import the built-in logging module and configure it so that Word2Vec
# creates nice output messages
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', \
                    level=logging.INFO)

num_features = 500  # Word vector dimensionality
min_word_count = 40  # Minimum word count
num_workers = 4  # Number of threads to run in parallel
context = 10  # Context window size
downsampling = 1e-3  # Downsample setting for frequent words

# Initialize and train the model (this will take some time)
from gensim.models import word2vec

print("Training model...")
model = word2vec.Word2Vec(sentences, workers=num_workers, \
                          size=num_features, min_count=min_word_count, \
                          window=context, sample=downsampling)
model_name = "500features_33swords_50000context"
model.save(model_name)
model.doesnt_match("man woman child kitchen".split())
model.doesnt_match("good nice perfect bad".split())

model.doesnt_match("good nice perfect bad".split())
model.most_similar("good")
model.most_similar("bad")
model.most_similar("awful")
model.most_similar("chinese")
model.most_similar("lol")
# load model
from gensim.models import Word2Vec

model = Word2Vec.load("500features_33swords_50000context")
model.wv.syn0.shape


def makeFeatureVec(words, model, num_features):
    # Function to average all of the word vectors in a given
    # paragraph
    #
    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,), dtype="float32")
    #
    nwords = 0.
    #
    # Index2word is a list that contains the names of the words in
    # the model's vocabulary. Convert it to a set, for speed
    index2word_set = set(model.wv.index2word)
    #
    # Loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.
            featureVec = np.add(featureVec, model[word])
    #
    # Divide the result by the number of words to get the average
    featureVec = np.divide(featureVec, nwords)
    return featureVec


def getAvgFeatureVecs(reviews, model, num_features):
    # Given a set of reviews (each one a list of words), calculate
    # the average feature vector for each one and return a 2D numpy array
    #
    # Initialize a counter
    counter = 0
    #
    # Preallocate a 2D numpy array, for speed
    reviewFeatureVecs = np.zeros((len(reviews), num_features), dtype="float32")
    #
    # Loop through the reviews
    for review in reviews:
        #
        # Print a status message every 1000th review
        if counter % 1000 == 0:
            print("Review %d of %d" % (counter, len(reviews)))
        #
        # Call the function (defined above) that makes average feature vectors
        reviewFeatureVecs[counter] = makeFeatureVec(review, model, \
                                                    num_features)
        #
        # Increment the counter
        counter = counter + 1
    return reviewFeatureVecs


# Calculate average feature vectors for training and testing sets,
# using the functions we defined above. Notice that we now use stop word
# removal.


clean_train_reviews = []
for i in range(train_num):
    clean_train_reviews.append(review_to_wordlist(df.iloc[i, 2], remove_stopwords=True))
trainDataVecs = getAvgFeatureVecs(clean_train_reviews, model, num_features)

test = df.iloc[train_num + 1:train_num + 501, 2]
clean_test_reviews = []
for i in range(len(test)):
    clean_test_reviews.append(review_to_wordlist(test.iloc[i], \
                                                 remove_stopwords=True))
testDataVecs = getAvgFeatureVecs(clean_test_reviews, model, num_features)
forest = RandomForestClassifier(n_estimators=100)
print("Fitting a random forest to labeled training data...")
# 其中有一部分有问题，可能是因为有翻译问题,或者删掉
forest = forest.fit(trainDataVecs[25000:50000], df.iloc[25000:train_num, 0])
result = forest.predict(testDataVecs)
output = pd.DataFrame(data={"id": df.iloc[train_num + 1:train_num + 501, 0], "sentiment": result})

correct = 0
for i in range(500):
    x = np.square(output.iloc[i, 0] - output.iloc[i, 1])
    correct += x
mse = correct / 500
mse  # 1.15
