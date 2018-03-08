import pandas as pd
import numpy as np

df = pd.read_csv('train_data.csv')
df_cat = df['categories']

# name category matrix columns
cat_name = pd.read_table('cat_name.txt', header=None)
cat_names = cat_name.ix[:, 0]  # transform to series
cat_mat = pd.DataFrame(columns=cat_names, index=range(len(df)), dtype="int32")

# fill in matrix
for i in range(len(df)):
    for j in range(0, len(cat_mat.columns)):
        if cat_mat.columns[j] in eval(df_cat[i]):
            cat_mat.loc[i][j] = 1
            print(i)
        else:
            cat_mat.loc[i][j] = 0

cat_mat.to_csv("category matrix.csv")

#for i in range(len(cat_mat)):
#    if (cat_mat.iloc[i, :].sum() + 1) == len(eval(df_cat[i])):
#       z = 1
#    else:
#        print(i)

#extra = []
#for i in range(len(cat_names)):
#    if cat_mat.iloc[:, i].sum() == 0:
#        extra.append(cat_names[i])
#        print(i)
#extra_col = pd.Series(extra)
#cat_mat = cat_mat.drop(extra_col, axis=1)

#cat_mat.to_csv("category matrix.csv")
#cat_mat = pd.read_csv('category matrix.csv', dtype="int32")
