# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 19:05:01 2018

@author: rqz
"""
import re
import pandas as pd  
df=pd.read_csv("train_data.csv")     
#import nltk
from nltk.corpus import stopwords 

###########################################METHOD1###############################
def review_to_words( raw_review ):
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
    return( " ".join( meaningful_words ))  
# Initialize an empty list to hold the clean reviews
clean_train_reviews = []

# Loop over each review; create an index i that goes from 0 to the length
# of the movie review list 
train_num=50000
for i in range(train_num):
    # Call our function for each one, and add the result to the list of
    # clean reviews
    if( (i+1)%1000 == 0 ):
        print ("Review %d of %d\n" % ( i+1, train_num ))
    clean_train_reviews.append( review_to_words( df.iloc[i,2] ) )   
##to bag of words
print ("Creating the bag of words...\n")
from sklearn.feature_extraction.text import CountVectorizer

# Initialize the "CountVectorizer" object, which is scikit-learn's
# bag of words tool.  
vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 7000) 
#怎么多加停词

train_data_features = vectorizer.fit_transform(clean_train_reviews)

# Numpy arrays are easy to work with, so convert the result to an 
# array
train_data_features = train_data_features.toarray()    
    
print(train_data_features.shape)
vocab = vectorizer.get_feature_names()#按字母排列顺序的
import numpy as np

# Sum up the counts of each vocabulary word
dist = np.sum(train_data_features, axis=0)

# For each, print the vocabulary word and the number of times it 
# appears in the training set
for tag, count in zip(vocab, dist):
    print (count, tag)
    
from sklearn.ensemble import RandomForestClassifier

# Initialize a Random Forest classifier with 100 trees
forest = RandomForestClassifier(n_estimators = 100) 

# Fit the forest to the training set, using the bag of words as 
# features and the sentiment labels as the response variable
#
# This may take a few minutes to run
forest = forest.fit( train_data_features, df.iloc[0:train_num,0])    
    
test = df.iloc[train_num+1:train_num+501,2]

print (test.shape)

# Create an empty list and append the clean reviews one by one
num_reviews = len(test)
clean_test_reviews = [] 

print ("Cleaning and parsing the test set movie reviews...\n")
for i in range(num_reviews):
    if( (i+1) % 1000 == 0 ):
        print ("Review %d of %d\n" % (i+1, num_reviews))
    clean_review = review_to_words( test.iloc[i] )
    clean_test_reviews.append( clean_review )

# Get a bag of words for the test set, and convert to a numpy array
test_data_features = vectorizer.transform(clean_test_reviews)
test_data_features = test_data_features.toarray()

# Use the random forest to make sentiment label predictions
result = forest.predict(test_data_features)

true_test=df.iloc[train_num+1:train_num+501,0]
accuracy=0
for i in range(500):
    x=np.square(result[i,0]-true_test[i,1])
    accuracy+=x
accuracy/500 
###########################################METHOD2###############################
  
#Part two for word2vec
#need to fix problems of  other language
import nltk

def review_to_wordlist( review, remove_stopwords=False ):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    #
    # 1. Remove HTML
    #  
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
    
new_data1=df.iloc[:170000,:]   
    
sentences = []  # Initialize an empty list of sentences

print ("Parsing sentences from training set")
for i in range(train_num):
    sentences += review_to_sentences(df.iloc[i,2], tokenizer)
# Import the built-in logging module and configure it so that Word2Vec 
# creates nice output messages
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)

num_features = 500   # Word vector dimensionality                      
min_word_count = 40   # Minimum word count                        
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size                                                                                    
downsampling = 1e-3   # Downsample setting for frequent words

# Initialize and train the model (this will take some time)
from gensim.models import word2vec
print ("Training model...")
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
# Calculate average feature vectors for training and testing sets,
# using the functions we defined above. Notice that we now use stop word
# removal.











clean_train_reviews = []
for i in range(train_num):
    clean_train_reviews.append(review_to_wordlist(df.iloc[i,2], remove_stopwords=True))
trainDataVecs = getAvgFeatureVecs( clean_train_reviews, model, num_features )
   
test = df.iloc[train_num+1:train_num+501,2]    
clean_test_reviews = []
for i in range(len(test)):
    clean_test_reviews.append( review_to_wordlist( test.iloc[i], \
        remove_stopwords=True ))
testDataVecs = getAvgFeatureVecs( clean_test_reviews, model, num_features )   
forest = RandomForestClassifier( n_estimators = 100 )    
print ("Fitting a random forest to labeled training data...")
#其中有一部分有问题，可能是因为有翻译问题,或者删掉 
forest = RandomForestClassifier(n_estimators = 100) 
forest = forest.fit( trainDataVecs[25000:50000], df.iloc[25000:train_num,0] )    
result = forest.predict( testDataVecs )    
output = pd.DataFrame( data={"id":df.iloc[train_num+1:train_num+501,0], "sentiment":result} )    

correct=0
for i in range(500):
    x=np.square(output.iloc[i,0]-output.iloc[i,1])
    correct+=x
mse=np.sqrt(correct/500)
mse#1.15

#######################################METHOD3#######################################
#kemans
from sklearn.cluster import KMeans
import time
word_vectors = model.wv.syn0
num_clusters = word_vectors.shape[0] / 10
num_clusters=10
# Initalize a k-means object and use it to extract centroids
kmeans_clustering = KMeans( n_clusters = num_clusters )
start = time.time() # Start time
idx = kmeans_clustering.fit_predict(word_vectors)

# Get the end time and print how long the process took
end = time.time()
elapsed = end - start
print ("Time taken for K Means clustering: ", elapsed, "seconds.")
# Create a Word / Index dictionary, mapping each vocabulary word to
# a cluster number  
word_centroid_map = dict(zip( model.wv.index2word, idx ))
for cluster in range(num_clusters):
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
train_centroids = np.zeros( (len(train), num_clusters), \
    dtype="float32" )

# Transform the training set reviews into bags of centroids
counter = 0
for review in clean_train_reviews:
    train_centroids[counter] = create_bag_of_centroids( review, \
        word_centroid_map )
    counter += 1

# Repeat for test reviews 
test_centroids = np.zeros(( len(test), num_clusters), \
    dtype="float32" )

counter = 0
for review in clean_test_reviews:
    test_centroids[counter] = create_bag_of_centroids( review, \
        word_centroid_map )
    counter += 1
from sklearn.linear_model import LinearRegression
lm = LinearRegression(fit_intercept=True, normalize=True, copy_X=True, n_jobs=1)
lm_fit = lm.fit(train_centroids,train.iloc[:,0])
lm_predict = lm_fit.predict(test_centroids)
b=mean_squared_error(test.iloc[:,0], lm_predict)
np.sqrt(b)













forest = RandomForestClassifier(n_estimators = 100)

# Fitting the forest may take a few minutes
print ("Fitting a random forest to labeled training data...")
forest = forest.fit(train_centroids, df.iloc[0:train_num,0])
result = forest.predict(test_centroids)

# Write the test results 
output = pd.DataFrame( data={"id":df.iloc[train_num+1:train_num+501,0], "sentiment":result} )    
correct=0
for i in range(500):
    x=np.square(output.iloc[i,0]-output.iloc[i,1])
    correct+=x
mse=correct/500
mse#1.056



##################################################METHOD4#################
from gensim import corpora, models
import gensim
#use sentence information
dictionary = corpora.Dictionary(sentences)
corpus = [dictionary.doc2bow(text) for text in sentences]
#corpus should like that
print(corpus[0])
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=5, id2word = dictionary, passes=20)
print(ldamodel.print_topics(num_topics=3, num_words=3))
#对于模型来讲，不是很有用，相当于计算了一个变量属于某个主题的概率和参数
#'-0.340 * "category" + 0.298 * "$M$" + 0.183 * "algebra" + -0.174 * "functor" + -0.168 * "operator"'



forest = RandomForestClassifier(n_estimators = 100) 
forest = forest.fit( X_train, y_train )    
result = forest.predict( X_test )    
output = pd.DataFrame( data={"id":Y_train, "sentiment":result} )    












