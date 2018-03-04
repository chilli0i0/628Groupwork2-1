# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 20:20:50 2018

@author: rqz
"""

import pandas as pd
#改下文件名
df=pd.read_csv('/Users/yilixia/Downloads/raw_test_ylxia.csv')
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
#剩下的中文手动翻译好了之后 输出      
df.to_csv('/Users/yilixia/Downloads/translation_lyi.csv',index=False,encoding='utf-8')




