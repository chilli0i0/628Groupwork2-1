# Yelp Comments Prediction

Hanmo Li| Qizheng Ren| Jiacheng Xu|Lixia Yi
--------|------------|------------|---------
ddd@wisc.edu|qren25@wisc.edu|abc@wisc.edu|kkk@wisc.edu


# Contents
* Introduction
* Code
* Plot
* Ipython Notebook
* Reference

# Introduction
This project is the second module of UW-Madison Spring2018-STAT628.
The aim is to predict the ratings of Yelp Data.
# Code
There are four documents in this part:
    * Hanmo Li contains:
    * Lixia Yi contains:
    * qizheng ren contains:
        1. Translation for train and test set.
        2. Word2vec have four feature creation models.
        3. NB_bayes+Xgboost contains the final model.
        4. Output_kaggle transform the result into kaggle submission file.
    * xjc contains:
# Plot
    * WordCloud Plot
    * XGBoost feature Importance Plot
# Ipython Notebook
The ipython notebook gives the summary of all the related works.
PPT1 is for the first week's presentation.
PPt2 is for the second week's presentation.
# Reference
未完待续




    1.	Basic data information. (mean, counts, histogram, sd, good words detection for some of the variables)
    2.	Select observations (cities listed in guidelines). Observations down to 840000.
    3.	Delete latitudes and longitudes since restaurant name will give the same information.
    4.	Translation. Use python Google translation API to change other language into English.
# Part2: (data cleaning)
    1.	For ‘text’ part:
        (1)	Use stop words table to separate the text and then add new things into stop words list. (By using NLTK) 
        (2)	Separate the text by NLTK and pick up all the adj. and non. into 2 sparse table.
        (3)	Dimensionality reduction for adj. text matrix, select a good threshold by using Chi-square test or PCA.

    2.	For categories part:
        First create a categories list and then select big categories rather than small categories if both exist. Transform it into a one-hot matrix.
    3.	For time part:
        Not yet discussed.
    4.	For city part: create a one-hot matrix
# Part3: (Model selection)
    SVM, Random Forest, XGBoost,(sklearn) gbdt( gradient boost decision tree) guide, LSTM.
# Part4: parameters tuning and model ensemble
