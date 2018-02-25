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

Refer:
* `Code/HanmoLi/data_clean.py`
* `Code/word2vec.py`