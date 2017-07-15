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

#==============================================================================
# data loading
#==============================================================================
# load training set
train = pd.read_csv('../data/train_2016_v2.csv')
train['transactiondate'] = pd.to_datetime(train['transactiondate'])



# subset feature set
parcelid = list(train['parcelid'])
feature = pd.read_csv('../data/properties_2016_subset.csv')
feature = feature.ix[feature['parcelid'].isin(parcelid)]
#feature.to_csv('../data/properties_2016_subset.csv', index=False)


#==============================================================================
# autocorrelation check
#=============================================================================

# explore time series model
logerror = train.groupby(['transactiondate'])['logerror'].mean()   
lb.pacf_plot(logerror)


#==============================================================================
# calendar date value treatment
#=============================================================================
# add month into feature set
train['month'] = train['transactiondate'].apply(lambda x: x.month)
train = pd.merge(feature, train, on='parcelid', how='right')
# calindar variable to be converted to years
train['yearbuilt'] = train['yearbuilt'].apply(lambda x: 2016-x)



#==============================================================================
# missing value treatment
#=============================================================================
# check missing value
na=lb.na_check(train)


# number of values count in each variable
value_count = lb.value_count(train)
cat = value_count.ix[value_count['value']<100]


#boolean variables needed to fill in missing values only
na_vars = ['poolcnt','pooltypeid2','pooltypeid10','pooltypeid7',\
'fullbathcnt','garagecarcnt','roomcnt','bedroomcnt']

for var in na_vars:
    train[var].fillna(0, inplace=True)

    
# boolean variables needed to fill in missing values and change to boolean type   
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

#check and renmove na
#variables with more than 20% na: throw away
#variables with less than 20% na: fill na with sample average
na=lb.na_check(train)
train = train.ix[:,train.columns.isin(list(na.ix[na['value']>0.8,'name']))]
train = train.dropna()
cols = train.columns
to_be_removed = ['landtaxvaluedollarcnt','structuretaxvaluedollarcnt',\
'taxvaluedollarcnt']
for col in cols:
     if len(train.ix[pd.isnull(train[col]),col]) > 0:
         print(col)
         if col in to_be_removed:
             train = train.ix[:,~pd.isnull(train[col])]
         else:
             train.ix[pd.isnull(train[col]),col] = np.mean(train[col])
 
             
#==============================================================================
# data visualization
#==============================================================================
# feature against dependent variable plot
#lb.feature_plot(train,'transactiondate','logerror')

             
# scale variables


#==============================================================================
# data paritition
#==============================================================================
# sperate training set to be two parts, for training and validation each
train_x,train_y,test_x,test_y = lb.data_partition(train,min(train['transactiondate']),\
min(train['transactiondate'])+dt.timedelta(days=90),\
min(train['transactiondate'])+dt.timedelta(days=70),'transactiondate','logerror')


#==============================================================================
# OLS
#==============================================================================
pred = lb.regression(train_x,train_y,test_x,test_y)
#lb.compare_models(test_y, pred)

#==============================================================================
# Random Forest
#==============================================================================

# random forest parameters to be optimized: n_trees, max_depth, max_features
train_x,train_y,vld_x,vld_y = lb.data_partition(train,min(train['transactiondate']),\
min(train['transactiondate'])+dt.timedelta(days=70),\
min(train['transactiondate'])+dt.timedelta(days=50),\
'transactiondate','logerror')

param_grid={"n_trees":[100],"max_depth":[4,6,8],"max_features":['sqrt',0.6,0.8]} 
#params=lb.gridsearch(lb.rf,param_grid,train_x,\
#                     train_y,test_x,test_y)
#print(params.iloc[0,1])





 


