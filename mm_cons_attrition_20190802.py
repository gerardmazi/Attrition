# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 15:03:42 2019

@author: gmazi
"""
import pandas as pd
from sklearn.model_selection import train_test_split

# Load consumer money market data
mm_raw = pd.read_csv('mm_in.csv')

# Stratified random sample by customer ID
mm = mm_raw.groupby('SK_PrimaryCustomerID').apply(lambda x: x.sample(n=1)).reset_index(drop=True)