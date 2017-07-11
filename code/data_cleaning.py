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


# feature against dependent variable plot
lb.feature_plot(train,'transactiondate','logerror')

# parce catgorical variables
value_count = lb.value_count(train)
cat = value_count.ix[value_count['value']<100]


# catgorical variables needed to fill in missing values only
na_vars = ['poolcnt','pooltypeid2','pooltypeid10','pooltypeid7',\
'fullbathcnt','garagecarcnt','roomcnt','bedroomcnt']

for var in na_vars:
    train[var].fillna(0, inplace=True)


train['fireplaceflag'].fillna(0, inplace=True)
train.ix[train['fireplaceflag']!=0, 'fireplaceflag'] = 1
train['fireplaceflag'] = train['fireplaceflag'].astype(float)
train['hashottuborspa'].fillna(0, inplace=True)
train.ix[train['hashottuborspa']!=0, 'hashottuborspa'] = 1
train['hashottuborspa'] = train['hashottuborspa'].astype(float)
train['taxdelinquencyflag'].fillna(0, inplace=True)
train.ix[train['taxdelinquencyflag']!=0, 'taxdelinquencyflag'] = 1
train['taxdelinquencyflag'] = train['taxdelinquencyflag'].astype(float)



# variables needed to convert to dummy variables
cats = ['storytypeid','decktypeid','buildingclasstypeid',\
'typeconstructiontypeid','regionidcounty','assessmentyear',\
'fips','numberofstories','threequarterbathnbr',\
'fireplacecnt','architecturalstyletypeid','airconditioningtypeid',\
'buildingqualitytypeid','unitcnt','finishedsquarefeet13',\
'taxdelinquencyyear','month','heatingorsystemtypeid',\
'propertylandusetypeid','propertycountylandusecode','regionidcity']

for cat in cats:
    train = lb.parse_dummy(train,cat)


# calindar variable to be converted to years
train['yearbuilt'] = train['yearbuilt'].apply(lambda x: 2016-x)


# variables not yet processed
to_be_processed = [
'regionidzip',
'regionidneighborhood',
'propertyzoningdesc',
'censustractandblock',
'rawcensustractandblock',
'longitude',
'latitude']

train = train.ix[:,list(set(train.columns).difference(to_be_processed))]

#check na
na=lb.na_check(train)

cols = train.columns
for col in cols:
    print(train.ix[pd.isnull(train[col]),col])
    
# scale variables

# sperate training set to be two parts, for training and validation each
train_x,train_y,test_x,test_y = lb.data_partition(train,min(train['transactiondate']),\
min(train['transactiondate'])+dt.timedelta(days=90),\
min(train['transactiondate'])+dt.timedelta(days=50),'transactiondate','logerror')


pred = lb.regression(train_x.ix[:,train_x.columns != 'transactiondate'],\
train_y,test_x.ix[:,test_x.columns != 'transactiondate'],test_y)






 


