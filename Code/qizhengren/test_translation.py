# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 20:05:31 2018

@author: rqz
"""

##do the test translation part
# we found the 339510th observation is wrong
import pandas as pd
test=pd.read_csv('testval_data.csv')
test.pop('categories')
test.pop("longitude")
test.pop('latitude')
test.pop('name')
test.pop('date')
test.pop('city')
df1=test.iloc[:400000,:]
df1.iloc[-1,0]
df2=test.iloc[400000:700000,:]
df3=test.iloc[700000:,:]
df2.to_csv('raw_test_ylxia.csv',index=False,encoding='utf-8')
df3.to_csv('raw_test_hmli.csv',index=False,encoding='utf-8')
df=df1[:]
from googletrans import Translator
from langdetect import detect
chinese=[]
other_lan=[]
for i in range(len(df)):
    if detect(df.iloc[i,1])=='ko' or detect(df.iloc[i,1])=='zh-tw':
        chinese.append(i)
    elif detect(df.iloc[i,1])!='en':
        translator = Translator()
        fake2=translator.translate(df.iloc[i,1])
        df.iloc[i,1]=fake2.text
        other_lan.append(i)
    else:
        print(i)

for i in chinese:
    translator = Translator()
    fake2=translator.translate(df.iloc[i,1])
    df.iloc[i,1]=fake2.text
result=df[:]    
test=pd.read_csv('testval_data.csv')   
result1=pd.read_csv('translation_lyi_test.csv')
result2=pd.read_csv('translated_II_hanmoli.csv')
data=pd.concat([result,result1,result2])
data.to_csv('fake.csv',index=False,header=True,encoding='utf-8')

#check the data's problem
'''
for i in range(len(xxx)):
    if xxx.iloc[i,0]=='All the goodness of the deep South! Seasoned and pan bronzed. Mississippi catfish served over Gouda grits with an andouille sausage and mushroom ragout.':
        print(i)
# data insight

for i in range(len(data)):
    if data.iloc[i,1]!=xxx.iloc[i,1]:
        print(i)


xxx=xxx.drop(339510)
for i in range(len(xxx)):
    if not isinstance(xxx.iloc[i,0],int):
        print(i,xxx.iloc[i,:])

a = [int(i) for i in xxx.iloc[:,0]]

'''

















