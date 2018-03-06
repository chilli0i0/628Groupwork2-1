# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 16:00:50 2018

@author: rqz
"""

#translation except for chinese fantizi
###yilixia range from 40,0001 to 80,0000
###xujiacheng range from 80,0001 to 12,0000
##lihanmo range from 12,0001 to the end (1546379)

## keep the chinese and other_lan id to translate later.
import numpy as np
import pandas as pd
df=pd.read_csv("train_data.csv")
from googletrans import Translator
from langdetect import detect
chinese=[]
other_lan=[]

#主要需要改一下这个range 在后面加数字就行
for i in range(400001):
    if detect(df.iloc[i,2])=='ko' or detect(df.iloc[i,2])=='zh-tw':
        chinese.append(i)
    elif detect(df.iloc[i,2])!='en':
        translator = Translator()
        fake2=translator.translate(df.iloc[i,2])
        df.iloc[i,2]=fake2.text
        other_lan.append(i)
    else:
        print(i)
        
for i in chinese:
    translator = Translator()
    fake2=translator.translate(df.iloc[i,2])
    df.iloc[i,2]=fake2.text
result=df.iloc[:400000,:]          

result.to_csv('translation_ren.csv',index=False,encoding='utf-8')
result1=pd.read_csv('translation_ren.csv')




