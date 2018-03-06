# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 18:58:26 2018

@author: rqz
"""
#read the predict result and transform to kaggle output
import pandas as pd
result=pd.read_csv("predict.txt")
new_data=pd.DataFrame(columns=['Id','Prediction1'])
new_data['Id'] = range(1,len(result)+1)
new_data['Prediction1']=list(result['result'])
new_data.to_csv('predict_new.csv',index=False,encoding='utf-8')