import pandas as pd
import string, re
import random
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import time

start = time.time()

df = pd.read_csv("/Users/yilixia/Downloads/stat628/train_translation.csv")
test = pd.read_csv("/Users/yilixia/Downloads/stat628/test_translation_new.csv")

sample_size = df.shape[0]

# shuffle the train set
sample = random.sample(range(df.shape[0]), sample_size)

# train = train.loc[:, ['stars', 'text']]
# test = test.loc[:, ['stars', 'text']]

label_cols = ['1', '2', '3', '4', '5']

thres = [0 + i*300000 for i in range(5)]
thres.append(-1)

est = []

# predict the stars using 5 different data sets
for i in range(5):
    # code chunk to keep track
    small_start = time.time()
    print(i+1, "loop")

    # use different set of train data
    new_sample = sample[thres[i]:thres[i+1]]
    train = df.loc[new_sample, ['stars', 'text']]

    # reform the stars from 1 column into 5 columns
    for j in label_cols:
        train[j] = (train['stars'] == int(j))


    re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')


    def tokenize(s): return re_tok.sub(r' \1 ', s).split()

    # n = df.shape[0]
    n = train.shape[0]
    # parameters are untuned!
    # term frequency–inverse document frequency
    vec = TfidfVectorizer(ngram_range=(1,2), tokenizer=tokenize,
                          min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,
                          smooth_idf=1, sublinear_tf=1)
    # This creates a sparse matrix with only a small number of non-zero elements
    trn_term_doc = vec.fit_transform(train['text'])
    test_term_doc = vec.transform(test['text'])  # fit a tfid sparse matrix with the same parameters as the train set's

    # the basic naive bayes feature equation, probability
    def pr(y_i, y):
        # y_i binary value {0,1}
        # y true value (whether the review is a 1-5 star review)
        p = x[y == y_i].sum(0)  # sum on axis 0 (column wise)
        # return normalized probability with laplace correction
        return (p+1) / ((y == y_i).sum()+1)

    x = trn_term_doc
    test_x = test_term_doc

    # get model
    def get_mdl(y):
        y = y.values  # array of the values
        r = np.log(pr(1, y) / pr(0, y))  # likelihood
        m = LogisticRegression(C=4, dual=True)  # logistic regression
        x_nb = x.multiply(r)
        # return a fitted model of x times its likelihood against y and the likelihood
        return m.fit(x_nb, y), r

    # predict test data

    # create a predict matrix (1-5 stars)
    preds = np.zeros((test.shape[0], len(label_cols)))

    for ii, jj in enumerate(label_cols):
        print('fit', jj)
        m, r = get_mdl(train[jj])  # feed the column indicating whether it is a 1-5 stars review into the model.
        print('predict', jj)
        # predict the probability of 1-5 stars
        preds[:, ii] = m.predict_proba(test_x.multiply(r))[:, 1]


    # predictions are the probabilities of stars (1-5)
    preds = np.asmatrix(preds)
    stars = np.matrix([1, 2, 3, 4, 5]).transpose()

    # multiply normalized probabilities with stars (Estimation)
    tmp_preds = preds.sum(1)
    tmp_preds = preds / tmp_preds
    tmp = tmp_preds * stars
    # append result to list
    est.append(tmp)

    small_end = time.time()
    print("time used in", i+1, "loop:", small_end - small_start)


est_matrix = np.transpose(est)

# our final prediction:
est_mean = est_matrix.mean(2)[0, ]

end = time.time()
print(end - start)
