# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 20:01:48 2018

@author: rqz
"""

import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
import time
t1 = time.time()
df=pd.read_csv("train_data.csv")
time.time() - t1
df.ix[:,4].value_counts()
#pick up the city we want
cities=["Edinburgh","Karlsruhe","Montreal","Waterloo","Pittsburgh","Charlotte","Urbana-Champaign","Phoenix","Las Vegas","Madison","Cleveland"]
colnames=df.columns.values.tolist()
new_data=pd.DataFrame(columns=[colnames])
for i in cities:
    new_data=pd.concat([new_data,df[df["city"]==i]],axis=0)
#observation detection
new_data.ix[:,1].value_counts()[new_data.ix[:,1].value_counts()>3]
delicious=new_data['stars'][new_data['text'].str.contains('delicious')]
each_word=new_data['stars'][new_data['text'].str.contains('delicious')]
np.sum(each_word==1)/len(each_word)


#text to txt
text=new_data.ix[:,2]
text.to_csv('reviews.txt',index=False,header=True)
all_text=pd.read_csv("reviews.txt")
# google translate
import goslate
with open('reviews_other_lan.txt', 'rb') as f:
    novel_text = f.read()
gs = goslate.Goslate()
reviews_other_lan=gs.translate(novel_text,"en")
#other language detection
from langdetect import detect
from langdetect import detect_langs 
detect_langs("Otec matka syn.") 
#other language column number
sum=0
other_lan=[]
for i in range(len(new_data)):
    if detect(new_data.iloc[i,2])!= 'en':
        print(i)
        sum+=1
        other_lan.append(i)
other_lan_pd=pd.Series(other_lan)
other_lan_pd.to_csv('other_language_columns.csv',header=False,index=False)
# other_lan=1673
other_lan_trans=[]
# read the other_language_columns.csv into other_lan
for i in range(len(other_lan)):
    gs = goslate.Goslate()
    fake=new_data.iloc[other_lan[i],2]
    result=gs.translate(fake, 'en')
    print(i)
    other_lan_trans.append(result)

# delete abonormal signs and simbols
import re
# Use regular expressions to do a find-and-replace
letters_only = re.sub("[^a-zA-Z]",           # The pattern to search for
                      " ",                   # The pattern to replace it with
                      example1.get_text() )  # The text to search
print (letters_only)



#tf-idf code
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer 
vectorizer = CountVectorizer()
corpus=df.iloc[0:10000,2]
X = vectorizer.fit_transform(corpus)
X.toarray() 
vectorizer.get_feature_names()
transformer = TfidfTransformer()



#pandas find data
####for some lines:
df.ix[1:10]
####for some columns:
df.ix[:,1:3]
### lines and columns with low efficiency
df.ix[0:10000,'text']
# name and position
df.loc[1:3,'text']
df.iloc[1:4,:2]
