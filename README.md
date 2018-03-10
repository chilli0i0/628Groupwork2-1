# Yelp Comments Prediction

Hanmo Li| Qizheng Ren| Jiacheng Xu|Lixia Yi
--------|------------|------------|---------
hli556@wisc.edu|qren25@wisc.edu|jxu354@wisc.edu|lyi28@wisc.edu


# Contents
* [Introduction](#Introduction)
* [Code](#Code)
* [Plot](#Plot)
* [Ipython Notebook](#I)


# <a id='Introduction'></a>Introduction
* This project is the second module of UW-Madison Spring2018-STAT628.
* The aim is to predict the ratings of Yelp Data.
* The data is from from Yelp DataSet Challenge (restaurant only), train set has over 150 million observations.
# <a id='Code'></a>Code
There are four documents in this part:
* **Hanmo Li contains:**

    1. data_clean: Tokenize the comments and creat the matrix of words frequency
    
    2. feature_seleciton: select features from reviews and other variables
    
    3. keras_LSTM: implementing the LSTM algorithm
    
    4. model_construction: conduct many machine learning algorithms using the sklearn package
    
    5. nbsvm: implement NBSVM algorithm and divid the algorithm into trainning and prediction parts
    
    6. vader_lexicon: emotion words and their scores ranging from -4 to 4
    
    7. word2vec: copy from qizheng ren's folder
    
* **Lixia Yi contains:**
    1. Testing on work of Hanmo Li and Qizheng Ren.
    2. Implementations of NBSVM
* **qizheng ren contains:**
    1. Translation for train and test set.
    2. Word2vec have four feature creation models.
    3. NB_bayes+Xgboost contains the final model.
    5. Parameters tuning contains the XGBoost tuning precedure.
    4. Output_kaggle transform the result into kaggle submission file.
* **xjc contains:**
# <a id='Plot'></a>Plot
* WordCloud Plot
* XGBoost Feature Importance Plot
# <a id='I'></a>Ipython Notebook
* The ipython notebook gives the summary of all the related works.
* PPT1 is for the first week's presentation.
* PPt2 is for the second week's presentation.

