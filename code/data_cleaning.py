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


# load feature set
feature = pd.read_csv('../data/properties_2016_subset.csv')
#feature = pd.read_csv('../data/properties_2016.csv')



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
feature['yearbuilt'] = feature['yearbuilt'].apply(lambda x: 2016-x)



#==============================================================================
# missing value treatment
#=============================================================================
# check missing value
na=lb.na_check(train, plot=True, name='training set')
na_test=lb.na_check(feature, plot=True, name='test set')



# number of values count in each variable
value_count = lb.value_count(train)
cat = value_count.ix[value_count['value']<100]



#boolean variables needed to fill in missing values only
na_vars = ['poolcnt','pooltypeid2','pooltypeid10','pooltypeid7',\
'fullbathcnt','garagecarcnt','roomcnt','bedroomcnt']

for var in na_vars:
    train[var].fillna(0, inplace=True)
    feature[var].fillna(0,inplace=True)


    
# boolean variables needed to fill in missing values and change to boolean type 
bl_vars = ['fireplaceflag','hashottuborspa','taxdelinquencyflag']
 
for var in bl_vars: 
    train[var].fillna(0, inplace=True)
    train.ix[train[var]!=0, var] = 1
    train[var] = train[var].astype(float)
    feature[var].fillna(0, inplace=True)
    feature.ix[feature[var]!=0, var] = 1
    feature[var] = feature[var].astype(float)


    
# variables needed to convert to dummy variables
cats = ['storytypeid','decktypeid','buildingclasstypeid',\
'typeconstructiontypeid','regionidcounty','assessmentyear',\
'fips','numberofstories','threequarterbathnbr',\
'fireplacecnt','architecturalstyletypeid','airconditioningtypeid',\
'buildingqualitytypeid','unitcnt',\
'taxdelinquencyyear','heatingorsystemtypeid',\
'propertylandusetypeid','propertycountylandusecode','regionidcity']

for cat in cats:
    train = lb.parse_dummy(train,cat)
    feature = lb.parse_dummy(feature,cat)

    
    
# process month variable    
train = lb.parse_dummy(train,'month')



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
feature = feature.ix[:,list(set(feature.columns).difference(to_be_processed))]




#check and renmove na
#variables with more than 20% na: throw away
#variables with less than 20% na: fill na with sample average
na=lb.na_check(train,name='training set with dummy')
train = train.ix[:,train.columns.isin(list(na.ix[na['value']>0.8,'name']))]
feature = feature.ix[:,feature.columns.isin(list(na.ix[na['value']>0.8,'name']))]


cols = train.columns
to_be_removed = ['landtaxvaluedollarcnt','structuretaxvaluedollarcnt',\
'taxvaluedollarcnt']
for col in cols:
     if len(train.ix[pd.isnull(train[col]),col]) > 0:
         if col in to_be_removed:
             del train[col]
         else:
             train.ix[pd.isnull(train[col]),col] = np.mean(train[col])

                         
cols = feature.columns
to_be_removed = ['landtaxvaluedollarcnt','structuretaxvaluedollarcnt',\
'taxvaluedollarcnt']
for col in cols:
     if len(feature.ix[pd.isnull(feature[col]),col]) > 0:
         if col in to_be_removed:
             feature = feature.ix[~pd.isnull(feature[col]),:]
         else:
             feature.ix[pd.isnull(feature[col]),col] = np.mean(feature[col]) 
             
             
             
             
#==============================================================================
# data visualization and scaling
#==============================================================================
# feature against dependent variable plot
#lb.feature_plot(train,'transactiondate','logerror')

 
            
# scale variables
cnts = lb.value_count(train)
to_be_scaled = list(cnts.ix[cnts['value']>2,'name'])
to_be_scaled.remove('parcelid')
to_be_scaled.remove('transactiondate')
to_be_scaled.remove('logerror')
train[to_be_scaled] = lb.scaler(train, to_be_scaled)




#==============================================================================
# dimension reduction and data partition
#==============================================================================
train_x,train_y,testx,testy = lb.data_partition(train,min(train['transactiondate']),\
max(train['transactiondate']),\
max(train['transactiondate']),'transactiondate','logerror')

lb.svd_figures(train_x,50)



train_x,train_y,test_x,test_y = lb.data_partition(train,min(train['transactiondate']),\
min(train['transactiondate'])+dt.timedelta(days=90),\
min(train['transactiondate'])+dt.timedelta(days=70),'transactiondate','logerror')

train_x,test_x = lb.dimension_reduction(train_x,test_x,50)



#==============================================================================
# OLS
#==============================================================================
pred = lb.regression(train_x,train_y,test_x)
lb.compare_models(test_y, pred,plot=False)




#==============================================================================
# Train Random Forest
#==============================================================================
# random forest parameters to be optimized: n_trees, max_depth, max_features

#train_x,train_y,vld_x,vld_y = lb.data_partition(train,min(train['transactiondate']),\
#min(train['transactiondate'])+dt.timedelta(days=70),\
#min(train['transactiondate'])+dt.timedelta(days=50),\
#'transactiondate','logerror')
#
#param_grid={"n_trees":[100],"max_depth":[4,6,8],"max_features":['sqrt',0.6,0.8]} 
#params=lb.gridsearch(lb.rf_train,param_grid,train_x,\
#                     train_y,vld_x,vld_y)
#print(params.iloc[0,1])
#
#pred = lb.rf_predict(train_x,train_y,test_x,n_trees=100,max_depth=8,\
#                     max_features='sqrt')
#lb.compare_models(test_y, pred,plot=False)




#==============================================================================
# output prediction on test set
#==============================================================================
feature['month_10'] = 1

train_x,train_y,testx,testy = lb.data_partition(train,min(train['transactiondate']),\
max(train['transactiondate']),\
max(train['transactiondate']),'transactiondate','logerror')

train_x,feature = lb.dimension_reduction(train_x,feature,50)

lg_pred = lb.regression(train_x,train_y,feature)

print('Random Forest Forecasting start...')
rf_pred = lb.rf_predict(train_x,train_y,feature,n_trees=100,max_depth=8,\
                max_features='sqrt')




