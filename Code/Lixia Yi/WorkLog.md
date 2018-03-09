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

-1

    This is Cantonese (Nightmare):
    今日好勇敢，叫咗個藜麥餅(Farmers Market Stack)做早餐！藜麥係非常有益嘅食物，點有益法可以自己去google睇下，以前當飯咁食過幾次，好難食，好強草味，好艱難先食得完！今次再試係對個廚師有期望，碟嘢賣相一流，食落有啲似蘿蔔糕，煮法係意式，所以係意式蘿蔔糕！好味！又有益！正！
    
-2    
    
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
    
-3    
    
    Extremly uninformative:
    潮州魚蛋粉!
    Chaozhou fish meal!
    (4 Star)
It may be deduced that the length of review is also important

### Feb 27

**Lexicon**: Extracted all the words with sentiment scores unequal to 0 (either pos or neg) as `SentiNetLexicon.txt`. Combined with the `vader_lexicon.txt`, they could be used as a reference for establishing the bag of words.

**Other_Lang** as a info source is alluring, however, doing a chi-square test shows that they do not differ significantly in the stars.

	Pearson's Chi-squared test

    data:  stars and other_stars
    X-squared = 20, df = 16, p-value = 0.2202
    
-----------a-n-o-t-h-e-r--t-e-s-t-----------
	
	Pearson's Chi-squared test

    data:  stars and chinese_stars
    X-squared = 15, df = 12, p-value = 0.2414
 
**Vader** 

Quote: "Manually creating and validating such lists of opinion-bearing features, while being among the most robust methods for generating reliable sentiment lexicons, is also one of the most time-consuming. For this reason, much of the applied research leveraging sentiment analysis relies heavily on preexisting manually constructed lexicons."


### March 1
**random forest based on lexicon**:

CountVectorizer has an option "input" where you can use a lexicon, but it wasn't successful

#### Suggestion for improvement: Expand the vader lexicon with the sentiNet lexicon even if it is not very appropriate.

**model_construction**: using only 5 variables (pos, neg, neu, comp, length)

* random forest: 0.934699339279717
* svm: 0.9626625198488145
* gradient boost: 0.9392548849453111
* naive bayes: 1.1403069762129845
* logistic: did not converge
* decision tree: 1.1773699503554522
* regression tree: 0.9482950239298016
* k-nearest: 1.2349898785010345
* k-nearest regression: 1.0493617107556397
* linear regression: 0.9738422761451919


### March 3
**NBSVM** :

* Source: https://www.kaggle.com/jhoward/nb-svm-strong-linear-baseline/notebook

Results are too good to believe:
```
* est.stars 0.6664561464259615
* prob*stars 0.9494745847706799
* max_prob:  0.7906563642008928
* time:  70.73907089233398
-------------------------
* est.stars 0.6244147329229339
* prob*stars 0.9248609143143568
* max_prob:  0.7380687809733285
* 155.96418404579163
```


```
lens = [len(x.split()) for x in df.text]
lens = pd.Series(lens)
df.loc[lens == lens.max(), ['stars', 'text']]
df.loc[lens == lens.min(), ['stars', 'text']]
lens.mean()
lens.var()
-------------------------
df.loc[lens == lens.min(), 'text']
Out[91]: 
140990    ok
387315    at
Name: text, dtype: object
df.loc[lens == lens.min(), 'stars']
Out[92]: 
140990    2
387315    5
Name: stars, dtype: int64
```

### March 4

In test data:
```
4059                                     Bland
32263                                 Average.
54805                            非常非常好吃，非常非常推荐
64334                                  Neither
67779                                   r.i.p.
76134                                     Yum!
149693                                   Blah.
199176                               美味しかったー！！
223042                                  Miamm!
226134                              Overrated.
228762                            Snickelfritz
369174                                 J'adore
376119                                Palabra!
382889                                 Decent.
427503                                Super!!!
445509                                  普通でした。
468450                              Excellent!
506109                                 closed!
516018                                    Bien
540971                                  r.i.p.
555996                                Correcte
673163                                  r.i.p.
731627                                   Bomb!
738464                                   mouai
748331                                   Gross
749039                               Hhhjjjhhh
756618                                 Monkey.
816660                                  closed
825752                                       X
959813    #notedible#doubleyuk!yuk!#neveragain
963217                                       E
971995                                       O

```
In train data:
```
203353                                                  Dont
210298                                                 Fast?
259204                                                     V
1104976                                                  RIP
1163069                                                   Wo
1199182                                               Vomit.
1373201    Spicyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy...
1391454                                              Poutine
1450257                                               r.i.p.
1475572                                                    y

```

### Mar 6

**IDEA**: Should we do multiple nb-svm and take mean? overlapping samples? ensembling methods?

I ran the NB-SVM method five times and take the mean using the whole dataset splitting it into 6 splits.
I used one split as the testing data (with 359784 rows of data) while the 5 others serve as training sets (each with about 200,000 rows of data).
It takes about 30 minutes to run the whole process but the results are quite promising

```
* est.stars 0.6157957335660347
* prob*stars 0.8858112279495591
* max_prob:  0.6605802881697144
```

### Mar 8
```
vocab = vec.get_feature_names()
'a' in vocab  # TRUE
'an' in vocab  # TRUE
'these' in vocab  # TRUE
'the' in vocab  # FALSE
```