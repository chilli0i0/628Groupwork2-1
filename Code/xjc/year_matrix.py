import pandas as pd
import datetime



df_1 = pd.read_csv("../train_data.csv")
date =  pd.DatetimeIndex(df_1['date']).year.values


year_names = list(range(2004,2017))
year_mat = pd.DataFrame(columns=year_names, index=range(len(date)), dtype="int32")

# fill in matrix
for i in year_names:
    year_mat[i] = (date == int(i))

year_mat.to_csv("time matrix.csv",index=False)

df = pd.read_csv("../testval_data.csv")
date1 =  pd.DatetimeIndex(df['date']).year.values


year_names = list(range(2004,2017))
year_mat1 = pd.DataFrame(columns=year_names, index=range(len(date1)), dtype="int32")

# fill in matrix
for i in year_names:
    year_mat1[i] = (date1 == int(i))

year_mat1.to_csv("time matrix for test.csv",index=False)
