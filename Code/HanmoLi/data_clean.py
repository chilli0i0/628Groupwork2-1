# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 20:09:03 2018

@author: lihan
"""



#import required packages
#basics
import pandas as pd 
import numpy as np





#nlp
import string
import re    #for regex
import nltk
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer 
from nltk.tokenize import word_tokenize
# Tweet tokenizer does not split at apostophes which is what we want
from nltk.tokenize import TweetTokenizer   


#FeatureEngineering
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

df=pd.read_csv("test_translation_new.csv")
df = df.drop(339510)
#df = df.drop(23794)
#cities=["Edinburgh","Karlsruhe","Montreal","Waterloo","Pittsburgh","Charlotte","Urbana-Champaign","Phoenix","Las Vegas","Madison","Cleveland"]
#colnames=df.columns.values.tolist()
#new_data=pd.DataFrame(columns=[colnames])
#for i in cities:
#    new_data=pd.concat([new_data,df[df["city"]==i]],axis=1)

#df["stars"][df["name"].str.contains("McDonald's")].hist()
#df["stars"][df["text"].str.contains("minutes")].mean()
#df["stars"][df["text"].str.contains("minutes")].hist()
#df["text"][df["stars"]==1].str.len().mean()
#df["text"].to_csv(r'C:\Users\lihan\OneDrive - UW-Madison\window\STAT 628\Module_2\reviews.txt')
df["upper_letters"]=[ sum(1 for c in i if c.isupper())/len(i) for i in df['text']]
df["upper_letters"][[i for i in range(df.shape[0]) if df['stars'][i]==1]].hist()

## Indirect features
stoplist = ['a', 'able', 'about', 'across', 'after', 'all', 'almost', 'also', 'am', 'among', 'an', 'and', 'any',
                  'are', 'as', 'at', 'be', 'because', 'been', 'by', 'did', 'else', 'ever', 'every', 'for', 'from',
                  'get', 'got', 'had', 'has', 'have', 'he', 'her', 'hers', 'him', 'his', 'how', 'however', 'i', 'if',
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
#Sentense count in each comment:
    #  '\n' can be used to count the number of sentences in each comment
df_simple_features = pd.DataFrame()
df_simple_features['count_sent']=df["text"].apply(lambda x: len(re.findall("\.",str(x)))+1)
#Word count in each comment:
df_simple_features['count_word']=df["text"].apply(lambda x: len(str(x).split()))
#Unique word count
df_simple_features['count_unique_word']=df["text"].apply(lambda x: len(set(str(x).split())))
#Letter count
df_simple_features['count_letters']=df["text"].apply(lambda x: len(str(x)))
#punctuation count
df_simple_features["count_punctuations"] =df["text"].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))
#upper case words count
df_simple_features["count_words_upper"] = df["text"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
#title case words count
df_simple_features["count_words_title"] = df["text"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))
#Number of stopwords
df_simple_features["count_stopwords"] = df["text"].apply(lambda x: len([w for w in str(x).lower().split() if w in stoplist]))
#Average length of the words
df_simple_features["mean_word_len"] = df["text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
#derived features
#Word count percent in each comment:
df_simple_features['word_unique_percent']=df_simple_features['count_unique_word']*100/df_simple_features['count_word']
#derived features
#Punct percent in each comment:
df_simple_features['punct_percent']=df_simple_features['count_punctuations']*100/df_simple_features['count_word']

df_simple_features.to_csv('simple_features_test.csv',index=False,encoding="utf-8")


# reviews clean and key words selection

from rake_nltk import Rake
import nltk
import string

stopset = stop_words = {'a', 'able', 'about', 'across', 'after', 'all', 'almost', 'also', 'am', 'among', 'an', 'and', 'any',
                  'are', 'as', 'at', 'be', 'because', 'been', 'by', 'did', 'else', 'ever', 'every', 'for', 'from',
                  'get', 'got', 'had', 'has', 'have', 'he', 'her', 'hers', 'him', 'his', 'how', 'however', 'i', 'if',
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
                  '$', '(', '.', ')'}
#stopset = set(nltk.corpus.stopwords.words('english'))
#stopset.remove("food")
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
    
#words = [i.lower() for i in word_tokenize(df["text"][12])]
#r = Rake(stopset,string.punctuation)
#sentence_re_stopwords = " ".join([token for token in words if token not in set(string.punctuation)])
#r.extract_keywords_from_text(sentence_re_stopwords)
#phrases = r.get_ranked_phrases()
#scores = r.get_ranked_phrases_with_scores()
#
# extract stars in the reviews:
#stars = list()
#pattern1 = re.compile('(\d+\.\d*) star')
#pattern2 = re.compile('(\d*) star')
#for i in range(1000):
#    words = [i.lower() for i in word_tokenize(df["text"][i])]
#    sentence_re_stopwords = " ".join([token for token in words if token not in set(string.punctuation)])
#    sen_re1 = pattern1.search(sentence_re_stopwords)
#    sen_re2 = pattern2.search(sentence_re_stopwords) 
#    if sen_re1 is None:
#        stars.append(0)
#        continue
#    else:
#        stars.append(float(sen_re1.group(1)))
#        continue
#    if sen_re2 is None:
#        stars.append(0)
#        continue
#    else:
#        stars.append(float(sen_re2.group(1)))
#
#[i for i in stars if i != 0]

# extract key words from sentences

r = Rake(stopset,string.punctuation)
all_keywords = list()
all_keywords_split = list()
train_num = 50000
print('starts to split reviews and extract key words...')
for i in range(train_num):
    if i%1000 == 0:
           print ("Review %d of %d" % (i, train_num))
    words = [j.lower() for j in nltk.word_tokenize(df.iloc[i,2])]
    sentence_re_stopwords = " ".join([token for token in words if token not in set(string.punctuation)])
    r.extract_keywords_from_text(sentence_re_stopwords)
    phrases = r.get_ranked_phrases()
    scores = r.get_ranked_phrases_with_scores()
    all_keywords.append([scores[j] for j in range(len(phrases)) if scores[j][0]>1])
    all_keywords_split.append([(scores[j][0],nltk.word_tokenize(phrases[j])) for j in range(len(phrases)) if scores[j][0]>1])

#import collections

all_keywords_each = list()
print('starts to calculate all_keywords_each...\n')
for  index,i in enumerate(all_keywords_split):
    if index%1000 == 0:
           print ("Review %d of %d" % (index, len(all_keywords_split)))
    for j in i:
        for l in range(len(j[1])):
            if j[1][l].isalpha():
                all_keywords_each.append(j[1][l])
            
keywords_in_each_review = list()
print('starts to calculate keywords_in_each_review...\n')
for  index,i in enumerate(all_keywords_split):
    if index%1000 == 0:
        print ("Review %d of %d" % (index, len(all_keywords_split)))
    tmp = list()
    score_list = [i[k][0] for k in range(len(i))]
    for index,j in enumerate(i):
        for l in range(len(j[1])):
            if j[1][l].isalpha():
                tmp.append((score_list[index],j[1][l]))
    keywords_in_each_review.append(tmp)

#collections.Counter(all_keywords_each).most_common()[0:99]



# extract emotion words from keywords

emotion_words = pd.read_csv("vader_lexicon.txt",sep = "\t",header=None)

emotion_words_list = emotion_words[0].tolist()
emotion_scores_list = emotion_words[1].tolist()

#emotion_words_times = [all_keywords_each.count(i) for i in emotion_words_list]
emotion_words_selected = list(filter(lambda x: x in all_keywords_each[0:100000],emotion_words_list))#[emotion_words_list[i] for i in range(len(emotion_words_list)) if emotion_words_list[i] in all_keywords_each]
emotion_related_scores = {emotion_words_list[i]:emotion_scores_list[i] for i in range(len(emotion_words_list)) if emotion_words_list[i] in emotion_words_selected}


emotion_words_in_reviews = list()
print('starts to calculate emotion_words_in_reviews...\n')
for index,i in enumerate(emotion_words_selected):
    print ("selected emotion words %d of %d" % (index, len(emotion_words_selected)))
    tmp = list()
    for j in range(len(keywords_in_each_review)):
        #if j%1000 == 0:
        #    print ("reviews %d of %d" % (j, len(keywords_in_each_review)))
        tmp.append([i[1] for i in keywords_in_each_review[j]].count(i))
    emotion_words_in_reviews.append(("critical_words"+str(index),tmp))

emotion_words_in_reviews_df = pd.DataFrame.from_items(emotion_words_in_reviews)

emotion_words_in_reviews_df.to_pickle('./features/emotion_words_in_reviews_df')


