# Hanmo/data_clean.py:

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

