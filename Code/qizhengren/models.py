# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 19:52:21 2018

@author: rqz
"""
import pandas as pd
import random
import numpy as np
import re
from nltk.corpus import stopwords

df=pd.read_csv('train_data.csv')
test=pd.read_csv('test_data.csv')


df1=pd.read_csv("translation_ren.csv") 
df1=df1.append(df.iloc[400000].to_frame().transpose()) 
df2=pd.read_csv("lyi_small.csv")
df2 = df2.drop(['a'],axis=1)
df2 = df2.drop(['b'],axis=1)
df2=df2.append(df.iloc[800000].to_frame().transpose()) 
df3=pd.read_csv('trans_xjc.csv')
df4=pd.read_csv("translation_hanmoli.csv")
data=pd.concat([df1,df2,df3,df4])
data.pop('categories')
data.pop("longitude")
data.pop('latitude')
data.pop('name')
data.pop('date')
data.pop('city')
data.to_csv('train_translation.csv',index=False,header=True,encoding='utf-8')
categories=pd.read_csv("categorymatrix.csv")
categories.pop('a')
random.seed(1)
fake=pd.concat([data,categories],axis=1,join_axes=[data.index])
data_index=random.sample(range(fake.shape[0]),55000)
data_=fake.iloc[data_index,:]
train=data_.iloc[:50000,:]
test=data_.iloc[50000:,:]
train.to_csv('train_new_lati.csv',index=False,header=True,encoding='utf-8')
test.to_csv('test_new_lati.csv',index=False,header=True,encoding='utf-8')
train=pd.read_csv('train_new_cate.csv')
test=pd.read_csv('test_new_cate.csv')


##add a random turbulent
random_train=np.random.randn(50000)-0.5
small_int_train=0.001*random_train
train1=train[:]
train1.iloc[:,0]= train.iloc[:,0]+small_int_train
random_test=np.random.randn(5000)-0.5
small_int_test=0.001*random_test
test1=test[:]
test1.iloc[:,0]= test.iloc[:,0]+small_int_test

#word2vec
import nltk

def review_to_wordlist( review, remove_stopwords=False ):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.  
    # 2. Remove non-letters
    review_text = re.sub("[^a-zA-Z]"," ", review)
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
    return(words)
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
def review_to_sentences( review, tokenizer, remove_stopwords=False ):
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
            sentences.append( review_to_wordlist( raw_sentence, \
              remove_stopwords ))
    #
    # Return the list of sentences (each sentence is a list of words,
    # so this returns a list of lists
    return sentences    

sentences = []  # Initialize an empty list of sentences

for i in range(len(train)):
    sentences += review_to_sentences(train.iloc[i,2], tokenizer)
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)

num_features = 300  # Word vector dimensionality                      
min_word_count = 80   # Minimum word count                        
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size                                                                                    
downsampling = 1e-3   # Downsample setting for frequent words

# Initialize and train the model (this will take some time)
from gensim.models import word2vec

model = word2vec.Word2Vec(sentences, workers=num_workers, \
            size=num_features, min_count = min_word_count, \
            window = context, sample = downsampling)
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
#load model
from gensim.models import Word2Vec
model = Word2Vec.load("500features_33swords_50000context")
model.wv.syn0.shape    
       
def makeFeatureVec(words, model, num_features):
    # Function to average all of the word vectors in a given
    # paragraph
    #
    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,),dtype="float32")
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
            featureVec = np.add(featureVec,model[word])
    # 
    # Divide the result by the number of words to get the average
    featureVec = np.divide(featureVec,nwords)
    return featureVec

def getAvgFeatureVecs(reviews, model, num_features):
    # Given a set of reviews (each one a list of words), calculate 
    # the average feature vector for each one and return a 2D numpy array 
    # 
    # Initialize a counter
    counter = 0
    # 
    # Preallocate a 2D numpy array, for speed
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
    # 
    # Loop through the reviews
    for review in reviews:
       #
       # Print a status message every 1000th review
       if counter%1000 == 0:
           print ("Review %d of %d" % (counter, len(reviews)))
       # 
       # Call the function (defined above) that makes average feature vectors
       reviewFeatureVecs[counter] = makeFeatureVec(review, model, \
           num_features)
       #
       # Increment the counter
       counter = counter + 1
    return reviewFeatureVecs    
clean_train_reviews = []  
for i in range(len(train)):
    clean_train_reviews.append(review_to_wordlist(train.iloc[i,2], remove_stopwords=True))
trainDataVecs = getAvgFeatureVecs( clean_train_reviews, model, num_features )
train_fake=np.delete(trainDataVecs,23794,0) 
fake1=train[:]
train_y=fake1.drop([23794])
clean_test_reviews = []
for i in range(len(test)):
    clean_test_reviews.append( review_to_wordlist( test.iloc[i,2], \
        remove_stopwords=True ))
testDataVecs = getAvgFeatureVecs( clean_test_reviews, model, num_features )   
#单词只剩最有一个，怎么办

import xgboost as xgb
from xgboost.sklearn import XGBRegressor
from sklearn.grid_search import GridSearchCV
xgb1=xgb.XGBRegressor(seed=1)   
xgb1.fit(train_fake,train_y.iloc[:,0])
result1=xgb1.predict(testDataVecs)
from sklearn.metrics import mean_squared_error
a=mean_squared_error(test.iloc[:,0], result1)
np.sqrt(a)





xgb2=XGBRegressor(
        base_score=0.5,
        colsample_bylevel=1,
        colsample_bytree=1,
        gamma=0,
        max_depth=3,
        min_child_weight=1,
        learning_rate=0.1,
        max_delta_step=0,
        missing=None,
        n_estimators=100,
        nthread=-1,
        objective='reg:linear',
        reg_alpha=0,
        reg_lambda=1,
        scale_pos_weight=1,
        seed=1,
        subsample=1)
xgb2.fit(train_fake,train_y.iloc[:,0]-1)
result2=xgb2.predict(testDataVecs)
b=mean_squared_error(test.iloc[:,0], result2+1)
np.sqrt(b)

from sklearn.linear_model import LinearRegression
lm = LinearRegression(fit_intercept=True, normalize=True, copy_X=True, n_jobs=1)
lm_fit = lm.fit(train_fake,train_y.iloc[:,0])
lm_predict = lm_fit.predict(testDataVecs)
b=mean_squared_error(test.iloc[:,0], lm_predict)
np.sqrt(b)
#########################################


param_test1={
        'max_depth':[3,5,7,9],
        'min_child_weight':list(range(1,6,2))}
gsearch1=GridSearchCV(estimator=XGBRegressor(
        base_score=0.5,
        colsample_bylevel=1,
        colsample_bytree=1,
        gamma=0,
        learning_rate=0.1,
        max_delta_step=0,
        missing=None,
        n_estimators=500,
        nthread=-1,
        objective='multi:softmax',
        num_class=6,
        reg_alpha=0,
        reg_lambda=1,
        scale_pos_weight=1,
        seed=1,
        subsample=1),param_grid=param_test1,scoring='neg_mean_squared_error',cv=5)
gsearch1.fit(X_train1,y_train1)
gsearch1.grid_scores_,gsearch1.best_params_,gearch1.best_score_









#############kmeans for categories


from sklearn.cluster import KMeans
from collections import Counter
train_kmeans=train1.drop([23794])
train_cate=train_kmeans.iloc[:,5:]
kmeans_clustering = KMeans(n_clusters=5, random_state=0).fit(train_cate)
klabels = kmeans_clustering.labels_
Counter(klabels)
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
lb = preprocessing.LabelBinarizer()
lb.fit(klabels)
lb.classes_
cat=lb.transform(klabels)





#kmeans 这里


word_centroid_map = dict(zip( model.wv.index2word, idx ))
for cluster in range(10):
    #
    # Print the cluster number  
    print ("\nCluster %d" % cluster)
    #
    # Find all of the words for that cluster number, and print them out
    words = []
    for i in range(len(word_centroid_map.values())):
        if( list(word_centroid_map.values())[i] == cluster ):
            words.append(list(word_centroid_map.keys())[i])
    print (words)

def create_bag_of_centroids( wordlist, word_centroid_map ):
    #
    # The number of clusters is equal to the highest cluster index
    # in the word / centroid map
    num_centroids = max( word_centroid_map.values() ) + 1
    #
    # Pre-allocate the bag of centroids vector (for speed)
    bag_of_centroids = np.zeros( num_centroids, dtype="float32" )
    #
    # Loop over the words in the review. If the word is in the vocabulary,
    # find which cluster it belongs to, and increment that cluster count 
    # by one
    for word in wordlist:
        if word in word_centroid_map:
            index = word_centroid_map[word]
            bag_of_centroids[index] += 1
    #
    # Return the "bag of centroids"
    return bag_of_centroids
# Pre-allocate an array for the training set bags of centroids (for speed)
train_centroids = np.zeros( (train.iloc[0:50000,2].size, num_clusters), \
    dtype="float32" )

# Transform the training set reviews into bags of centroids
counter = 0
for review in clean_train_reviews:
    train_centroids[counter] = create_bag_of_centroids( review, \
        word_centroid_map )
    counter += 1

# Repeat for test reviews 
test_centroids = np.zeros(( df.iloc[50001:50501,2].size, num_clusters), \
    dtype="float32" )

counter = 0
for review in clean_test_reviews:
    test_centroids[counter] = create_bag_of_centroids( review, \
        word_centroid_map )
    counter += 1



























































