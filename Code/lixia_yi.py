#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 21:37:19 2018

@author: yilixia
"""

import pandas as pd
import numpy as np
import re

df = pd.read_csv('/Users/yilixia/Downloads/train_data.csv')
df.ix[:, 4].value_counts()
cities = ["Edinburgh", "Karlsruhe", "Montreal", "Waterloo", "Pittsburgh", "Charlotte", "Urbana-Champaign", "Phoenix", "Las Vegas", "Madison", "Cleveland"]
colnames = df.columns.values.tolist()
new_data = pd.DataFrame(columns=[colnames])
for i in cities:
    new_data = pd.concat([new_data, df[df["city"] == i]], axis=0)
    

