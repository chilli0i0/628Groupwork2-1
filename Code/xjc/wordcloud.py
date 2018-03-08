import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
import nltk

df = pd.read_csv("train_data.csv")

new_data = df[df['city'].isin(
    ['Edinburgh', 'Karlsruhe', 'Montreal', 'Waterloo', 'Pittsburgh', 'Charlotte', 'Urbana-Champaign', 'Phoenix',
     'Las Vegas', 'Madison', 'Cleveland'])]
new_data.ix[:, 4].value_counts()

from wordcloud import WordCloud


def plot_word_cloud(data):
    wordcloud = WordCloud(background_color="white", width=1280,height=720).generate(str(data))
    plt.figure(figsize=(15, 7.5), facecolor='k')
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()


type(new_data["text"].loc[new_data.stars == 1,])
plot_word_cloud(df["text"].loc[df.stars == 1,])
plot_word_cloud(df["text"].loc[df.stars == 5,])

import goslate

with open('reviews_other_lan.txt', 'rb') as f:
    novel_text = f.read()
gs = goslate.Goslate()
reviews_other_lan = gs.translate(novel_text, "en")

with open("translation.txt", "w") as textfile:
    print(reviews_other_lan, file=textfile)


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



other_lan_trans=[]
# read the other_language_columns.csv into other_lan
colnames = ['number']
other_lan = pd.read_csv("other_language_columns.csv", names=colnames)
number = other_lan.number.tolist()
for i in range(len(number)):
    gs = goslate.Goslate()
    fake = new_data.iloc[number[i], 2]
    result = gs.translate(fake, 'en')
    print(i)
    other_lan_trans.append(result)
