# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 15:32:47 2018

@author: lihan
"""

'''Trains a Bidirectional LSTM on the IMDB sentiment classification task.

Output after 4 epochs on CPU: ~0.8146
Time per epoch on CPU (Core i7): ~150s.
'''

from __future__ import print_function
import numpy as np

from keras.layers import Dense,Input,LSTM,Bidirectional,Activation,Conv1D,GRU
from keras.callbacks import Callback
from keras.utils import to_categorical
from keras.layers import Dropout,Embedding,GlobalMaxPooling1D, MaxPooling1D, Add, Flatten
from keras.preprocessing import text, sequence
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D,GlobalMaxPool1D, concatenate, SpatialDropout1D
from keras import initializers, regularizers, constraints, optimizers, layers, callbacks
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.models import Model
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

import pickle
import pandas as pd
import re
from sklearn.cross_validation import train_test_split
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
#from Word2VecUtility import Word2VecUtility
from nltk.stem import PorterStemmer
from keras.layers.core import  Activation, Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D


def review_to_wordlist( review, remove_stopwords=False ):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    #
    # Remove HTML
    review_text = BeautifulSoup(review).get_text()
    # Remove non-letters
    review_text = re.sub("[^a-zA-Z!?0-9]"," ", review_text)
    review_text = re.sub("[!]", " !", review_text)
    review_text = re.sub("[?]", " ?", review_text)
    # Convert words to lower case and split them
    words = review_text.lower().split()
    # Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    #Implement porter stemmer
    stemmer = PorterStemmer()
    words = [stemmer.stem(w) for w in words]
    # Return a list of words
    return(words)

def get_volcabulary_and_list_words(data):
    reviews_words = []
    volcabulary = []
    for review in data["text"]:
        review_words = review_to_wordlist(review, remove_stopwords=True)
        reviews_words.append(review_words)
        for word in review_words:
            volcabulary.append(word)
    volcabulary = set(volcabulary)
    return volcabulary, reviews_words

def get_reviews_word_index(reviews_words, volcabulary, max_words, max_length):
    word2index = {word: i for i, word in enumerate(volcabulary)}
    # use w in volcabulary to limit index within max_words
    reviews_words_index = [[start] + [(word2index[w] + index_from) for w in review] for review in reviews_words]
    # in word2vec embedding, use (i < max_words + index_from) because we need the exact index for each word, in order to map it to its vector. And then its max_words is 5003 instead of 5000.
    reviews_words_index = [[i if (i < max_words) else oov for i in index] for index in reviews_words_index]
    # padding with 0, each review has max_length now.
    reviews_words_index = sequence.pad_sequences(reviews_words_index, maxlen=max_length, padding='post', truncating='post')
    return reviews_words_index

def vectorize_labels(labels, nums):
    labels = np.asarray(labels, dtype='int32')
    length = len(labels)
    Y = np.zeros((length, nums))
    for i in range(length):
        Y[i, (labels[i]-1)] = 1.
    return Y

# data processing para
max_words = 37000
max_length = 70

# model training parameters
batch_size = 32
embedding_dims = 100
nb_filter = 250
filter_length = 3
hidden_dims = 250
nb_epoch = 2

# index trick parameters
index_from = 3
start = 1
# padding = 0
oov = 2


reviews = pd.read_csv('train_new_lati.csv', header=0, delimiter=",", encoding='utf-8')
print('get volcabulary...')
volcabulary, reviews_words = get_volcabulary_and_list_words(reviews)
print('get reviews_words_index...')
reviews_words_index = get_reviews_word_index(reviews_words, volcabulary, max_words, max_length)

print(reviews_words_index[:20,:12])
print(reviews_words_index.shape)

labels = reviews["stars"]
#labels[labels <= 3] = 0
#labels[labels > 3] = 1

pickle.dump((reviews_words_index, labels), open("399850by50reviews_words_index.pkl", 'wb'))

# with oov, index_from, start and padding, we have 4999 + 4 = 5003 indexes.
(reviews_words_index, labels) = pickle.load(open("399850by50reviews_words_index.pkl", 'rb'))

index = np.arange(reviews_words_index.shape[0])
train_index, valid_index = train_test_split(index, train_size=0.8, random_state=520)

labels = vectorize_labels(labels, 5)
train_data = reviews_words_index[train_index]
valid_data = reviews_words_index[valid_index]
train_labels = labels[train_index]
valid_labels = labels[valid_index]
print(train_data.shape)
print(valid_data.shape)
print(train_labels[:10])
del(labels, train_index, valid_index)

# get the embadding matrix
#texts = reviews["text"].values
#stars = np.asarray([i-1 for i in reviews["stars"].values])
#MAX_NUM_WORDS=max_words # how many unique words to use (i.e num rows in embedding vector)
#MAX_SEQUENCE_LENGTH=max_length # max number of words in a review to use
#
#
#tokenizer = text.Tokenizer(num_words=MAX_NUM_WORDS)
#tokenizer.fit_on_texts(texts)
#sequences = tokenizer.texts_to_sequences(texts)
#
#word_index = tokenizer.word_index
#print('Found %s unique tokens.' % len(word_index))
#
#data = sequence.pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
#
#labels = to_categorical(np.asarray(stars))
#print('Shape of data tensor:', data.shape)
#print('Shape of label tensor:', stars.shape)
#VALIDATION_SPLIT=0.2
#indices = np.arange(data.shape[0])
#np.random.shuffle(indices)
#data = data[indices]
#labels = labels[indices]
#nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])
#
#x_train = data[:-nb_validation_samples]
#y_train = labels[:-nb_validation_samples]
#x_test = data[-nb_validation_samples:]
#y_test = labels[-nb_validation_samples:]
#
#embeddings_index = {}
#f = open('glove.6B.50d.txt','rb')
#for line in f:
#    values = line.split()
#    word = values[0]
#    coefs = np.asarray(values[1:], dtype='float32')
#    embeddings_index[word] = coefs
#f.close()
#
#print('Found %s word vectors.' % len(embeddings_index))
#EMBEDDING_DIM = embedding_dims # how big is each word vector
#
#embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
#for word, i in word_index.items():
#    embedding_vector = embeddings_index.get(word)
#    if embedding_vector is not None:
#        # words not found in embedding index will be all-zeros.
#        embedding_matrix[i] = embedding_vector


# Bidirectional LSTM
#model = Sequential()
#model.add(Embedding(max_words + index_from, embedding_dims, \
#                    input_length=max_length))
# Bidirectional LSTM
#model.add(Bidirectional(LSTM(64)))
#model.add(Dropout(0.5))
# LSTM
#model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))


# LSTM + fasttext from kaggle https://www.kaggle.com/CVxTz/keras-bidirectional-lstm-baseline-lb-0-069/output
inp = Input(shape=(max_length, ))
x = Embedding(max_words + index_from, embedding_dims)(inp)
x = Bidirectional(LSTM(50, return_sequences=True))(x)

x = GlobalMaxPool1D()(x)
x = Dropout(0.1)(x)

x = Dense(50, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(5, activation="sigmoid")(x)
model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

#inp = Input(shape=(MAX_SEQUENCE_LENGTH,))
#x = embedded_sequences = embedding_layer(inp)
#x = Bidirectional(LSTM(50, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(x)
#x = GlobalMaxPool1D()(x)
#x = Dense(50, activation="relu")(x)
#x = Dropout(0.1)(x)
#x = Dense(5, activation="sigmoid")(x)
#model = Model(inputs=inp, outputs=x)
#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#model.add(Dense(5, activation='sigmoid'))

# try using different optimizers and different optimizer configs
#model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
batch_size = 128
epochs = 2
class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: {:d} - score: {:.6f}".format(epoch+1, score))
X_tra, X_val, y_tra, y_val = train_test_split(train_data, train_labels, train_size=0.9, random_state=233)
file_path="weights_base.best.hdf5"
checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
early = EarlyStopping(monitor="val_loss", mode="auto", patience=2)

ra_val = RocAucEvaluation(validation_data=(X_val, y_val), interval = 1)
callbacks_list = [ra_val,checkpoint, early] #early
print('Train...')
model.fit(X_tra, y_tra, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val),callbacks = callbacks_list,verbose=1)

classes = model.predict(valid_data)

import math

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

sum_to_1 = [[j/sum(i) for j in i] for i in classes]
classes_unique = [sum(map(lambda x,y:x*y,i,range(1,6))) for i in sum_to_1]
test_labels = []
for i in range(len(valid_labels)):
    tmp_list = list(valid_labels[i])
    test_labels.append(tmp_list.index(max(tmp_list))+1)

print("expection:",mse_calculation(test_labels,classes_unique))
 # 0.76
 
classes_max = [list(i).index(max(list(i)))+1 for i in classes]
print("Max:",mse_calculation(test_labels,classes_max))

