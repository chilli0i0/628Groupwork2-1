# Stat 628 Module 2
## Group Member: Lixia Yi 

### Feb 22:

##### Some useful libraries:

**"VADER-Sentiment-Analysis"**, a nice method included in the 'nltk' library.

Reference:
1. Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. Eighth International Conference on Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, June 2014.
2. https://github.com/cjhutto/vaderSentiment
3. http://www.nltk.org/howto/sentiment.html



Things worth mention:

* "If you have access to the Internet, the demo will also show how VADER can work with analyzing sentiment of non-English text sentences."


**python-rake**, A Python module implementation of the Rapid Automatic Keyword Extraction (RAKE) algorithm.

Reference:
1. Rose, S., Engel, D., Cramer, N., & Cowley, W. (2010). Automatic Keyword Extraction from Individual Documents. In M. W. Berry & J. Kogan (Eds.), Text Mining: Theory and Applications: John Wiley & Sons. 
2. https://github.com/fabianvf/python-rake
3. Initially by @aneesha, packaged by @tomaspinho.

##### Implementing vaderSentiment:

I am meeting problems in linear regression...


### Feb 25:
**"VADER-Sentiment-Analysis"** continue: Made a box-plot to demonstrate how well the sentiment scores are related to stars.
It was implemented on 10000 rows of data.

Refer: 
* `Code/Lixia/vaderSentiment_stars.png`
* `lixia_yi_nltk.py`
 
 
**Read HML's and QZR's work**: Read their code, understand what they did and try to improve on their basis or try new ways.

Hanmo-data_clean.py:
* *use nlty to remove stopwords first* -- this is just a test fragment,
* *extract stars in the reviews* -- since some of the reviews have scored the restaurants directly in their text, he attempts to extract the scores from the first 1000 reviews.
It turns out that only 6 reviews have scored directly in the text. The scores given, however, mostly are ending with a .5 and could be rounded either up and down.
It could be used as a checker if scores (e.g. predicted 2 when text says 3.5) fall to far away but cannot really be regarded as solid prediction.
* *extract key words from sentences* -- Extract keywords from the first 1000 rows.
* *extract emotion words from keywords* -- From the keywords extracted in the previous part, extract the emotion words.
The emotion words are based on the vaderSentiment lexicon. 
* *calculate four features* -- 
    * def calculate_relative_score(keywords_in_each_review, ith_review, emotion_word)
    * def calculate_identital_score(emotion_word, emotion_related_scores):
    * def calculate_sent_posi_score(keywords_in_each_review, ith_sentence, positive_words_selected, emotion_related_scores):
    * def calculate_sent_negative_score(keywords_in_each_review, ith_sentence, negative_words_selected, emotion_related_scores):


Refer:
* `Code/HanmoLi/data_clean.py`
* `Code/word2vec.py`

**Translate**: Since it is slow to let one person to translate all text, we separated the text into 4 parts, each one of us runs one part.
Doing it in the terminal.

Stopped at 741038: 'http://jentalkstoomuch.com/2016/08/portland-variety-insulting-people-one-pregnant-woman-at-a-time/'
which links to a site with lots of lots of words and is rate with 1 star, still could it be deduced from the link itself but we should be aware of noise like this.
//result.to_csv('name.csv',index=False)
 
 
### Feb 26
**QZR's work**:
 
Regarding the output fo line 76-77:
```
for tag, count in zip(vocab, dist):
    print (count, tag)
```

It shows that there are words with the same stems (similar meaning, different forms) regarded as different factors. This could be a suggestion for improvement.

* Try Lexicon Normalization
* Generalization
    
Furthermore, there are definitely languages other than English in the text, but this should be solved after we've translated all the text information.
But again, Chinese and Korean were not translated (if I'm not mistaken) and we should be aware of it.
--Translated them manually in lyi.csv

Some stopwords are definitely useful! Don't delete them all!


**Other ideas**: 

Our problem isn't about the models but about the data cleaning, that is, did we extracted the all, or most, useful information from the text data?


* t-SNE clustering
* Neural Network (if possible)

Other_lang and Chinese could be used as a label itself. As I've noticed in my data set, Chinese reviews are highly biased (out of 16 only one of it is 1 star and the rest are 4 or 5 stars) so are the reviews in other languages.
However, we should test is anyway.

**Class Notes**:

* Some stopwords may be informative! Don't delete them all.

* His, her, he, she tend to be negative informative.

* It could be possible that some dishes/words are more informative than others, however, I doubt whether the less informative words will still be not informative after doing a split (in a decision tree for example)

* "Yum!"

* Combinations of foods might be an interesting indicator for stars.

* How should we do interaction? Shouldn't it be integrated in a decision tree?

* Sentiment words definitely have a influence on stars...of course ("affordable")

**Translation**:

The non-english reviews are **heavily** biased!

Translate the Chinese and Korean reviews manually:

    This is Cantonese (Nightmare):
    今日好勇敢，叫咗個藜麥餅(Farmers Market Stack)做早餐！藜麥係非常有益嘅食物，點有益法可以自己去google睇下，以前當飯咁食過幾次，好難食，好強草味，好艱難先食得完！今次再試係對個廚師有期望，碟嘢賣相一流，食落有啲似蘿蔔糕，煮法係意式，所以係意式蘿蔔糕！好味！又有益！正！
    
    Usage of words hard to translate:
    在拉斯維加斯中所有的吃到飽裡面，小奴婢最喜歡這家店
    雖然這家店沒有特別光鮮的外表，也沒有華麗的擺設
    但是他有特別貼近人心的服務
    小奴婢有注意到，這家店的服務員年紀都偏大(當然還是有年輕的妹妹)
    而且各個感覺都是做很久的樣子
    用餐的感覺很舒服，比較沒有那麼商業
    剛好咱們那天是享用早餐，所以特別有感
    其他時段因為價錢不一樣，所以食物小奴婢就不知道了
    不過早餐來這邊吃小奴婢覺得很適合喔~~!!
    雖然說是早餐，但是還是有牛排的XDDD
    更別說是其他熟食了!
    不過早餐該出現的食物這邊也沒有少
    像是麵包、優格、麥片等等的喔~
    
    "小奴婢--small slave..."
    
    Extremly uninformative:
    潮州魚蛋粉!
    Chaozhou fish meal!
    (4 Star)
It may be deduced that the length of review is also important