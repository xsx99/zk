# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 14:16:00 2017

@author: Shuxin Xu, Tao Hong
"""

import pandas as pd
import numpy as np
import datetime as dt
import library as lb
import imp
imp.reload(lb)


# load training set
train = pd.read_csv('../data/train_2016_v2.csv')
train['transactiondate'] = pd.to_datetime(train['transactiondate'])



# subset feature set
parcelid = list(train['parcelid'])
feature = pd.read_csv('../data/properties_2016_subset.csv')
feature = feature.ix[feature['parcelid'].isin(parcelid)]
#feature.to_csv('../data/properties_2016_subset.csv', index=False)




# explore time series model
logerror = train.groupby(['transactiondate'])['logerror'].mean()   
lb.pacf_plot(logerror)



# add month into feature set
train['month'] = train['transactiondate'].apply(lambda x: x.month)
train = pd.merge(feature, train, on='parcelid', how='right')



# check missing value
na=lb.na_check(train)

# parce catgorical varibles

# scale variables

# sperate training set to be two parts, for training and validation each
train_x,train_y,test_x,test_y = lb.data_partition(train,min(train['transactiondate']),\
min(train['transactiondate'])+dt.timedelta(days=90),\
min(train['transactiondate'])+dt.timedelta(days=90),'transactiondate','logerror')








 


