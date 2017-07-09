# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 14:16:00 2017

@author: Shuxin Xu, Tao Hong
"""

import pandas as pd


# load training set
train = pd.read_csv('../data/train_2016_v2.csv')
train['transactiondate'] = pd.to_datetime(train['transactiondate'])


# subset feature set
parcelid = list(train['parcelid'])
feature = pd.read_csv('../data/properties_2016.csv')
feature = feature.ix[feature['parcelid'].isin(parcelid)]

# 