import pandas as pd

df = pd.read_csv("/Users/yilixia/Downloads/testval_data.csv")


# data insight
lens = [len(x.split()) for x in df.text]
lens = pd.Series(lens)
df.loc[lens == lens.max(), 'text']
df.loc[lens == lens.min(), 'text']
lens.mean()
lens.var()


df = pd.read_csv("/Users/yilixia/Downloads/train_data.csv")

lens = [len(x.split()) for x in df.text]
lens = pd.Series(lens)
df.loc[lens == lens.max(), ['stars', 'text']]
df.loc[lens == lens.min(), ['stars', 'text']]
lens.mean()
lens.var()
