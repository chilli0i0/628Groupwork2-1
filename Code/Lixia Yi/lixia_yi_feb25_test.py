# Hanmo/data_clean.py:
import numpy as np
import  matplotlib.pyplot as plt
import ggplot
# extract stars in the reviews:
stars = list()
index = []
pattern1 = re.compile('(\d+\.\d*) star')
pattern2 = re.compile('(\d*) star')
for i in range(1000):
    words = [i.lower() for i in word_tokenize(df["text"][i])]
    sentence_re_stopwords = " ".join([token for token in words if token not in set(string.punctuation)])
    sen_re1 = pattern1.search(sentence_re_stopwords)
    sen_re2 = pattern2.search(sentence_re_stopwords)
    if sen_re1 is None:
        stars.append(0)
        continue
    else:
        stars.append(float(sen_re1.group(1)))
        index.append(i)
        continue
    if sen_re2 is None:
        stars.append(0)
        continue
    else:
        stars.append(float(sen_re2.group(1)))
        index.append(i)

[i for i in stars if i != 0]
df.loc[index, 'stars']

df.loc[index[4], 'text']


# extract emotion words from keywords

emotion_words = pd.read_csv("Code/HanmoLi/vader_lexicon.txt",sep = "\t",header=None)




# translation.py
# run it in terminal

import pandas as pd
df = pd.read_csv("/Users/yilixia/Downloads/train_data.csv")
from googletrans import Translator
from langdetect import detect
chinese=[]
other_lan=[]
#主要需要改一下这个range 在后面加数字就行
for i in range(400001,800000):
    if detect(df.iloc[i,2])=='ko' or detect(df.iloc[i,2])=='zh-tw':
        chinese.append(i)
    elif detect(df.iloc[i,2])!='en':
        translator = Translator()
        fake2=translator.translate(df.iloc[i,2])
        df.iloc[i,2]=fake2.text
        other_lan.append(i)
    else:
        print(i)



df = pd.read_csv("/Users/yilixia/Downloads/lyi.csv")

# df1.iloc[400001:800000, :].to_csv("/Users/yilixia/Downloads/lyi_small.csv")

from googletrans import Translator
from langdetect import detect
chinese=[]
other_lan=[]
#主要需要改一下这个range 在后面加数字就行
for i in range(741039, 800000):
    if detect(df.iloc[i,2])=='ko' or detect(df.iloc[i,2])=='zh-tw':
        chinese.append(i)
    elif detect(df.iloc[i,2])!='en':
        translator = Translator()
        fake2=translator.translate(df.iloc[i,2])
        df.iloc[i,2]=fake2.text
        other_lan.append(i)
    else:
        print(i)

df.iloc[400001:800000, :].to_csv("/Users/yilixia/Downloads/lyi_small.csv")

# Translate Chinese reviews by hand...
chinese = [413847, 416008, 460112, 464195, 522060, 539005, 547678, 553796, 557479, 604962, 678612, 682042, 703100,
           722513, 740363, 745057, 789778]
chinese = [x - 400001 for x in chinese]

i = 16
print(df.loc[chinese[i], 'stars'])
print(df.loc[chinese[i], 'text'])

df.loc[chinese[i], 'text'] = "Mom patties \
beef patty \
Fried noodles \
Paste cavity"

other_lan = pd.read_csv("/Users/yilixia/Downloads/lyi_other_lan.txt", sep=", ", header=None)  # len = 3252
other_lan = [x - 400001 for x in other_lan.values]
other_lan = other_lan[0]
other_stars = df.loc[other_lan, 'stars']




