# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 17:08:40 2017

@author: Shuxin Xu
"""


import pandas as pd
import numpy as np
import datetime as dt

import matplotlib.pyplot as plt

import statsmodels.tsa.api as tsa
import statsmodels.formula.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std




def pacf_plot(var,I=0):
    
    ''' Plot partial autocorrelation function of the input time series
        var: pandas series
            input time series
        I: integer
            number of time differences to be performed on the time series
    '''
    
    while (I!=0):
        var=var-var.shift(1)
        I=I-1
    var=var.ix[~pd.isnull(var)]
    tsa.graphics.plot_pacf(var,lags=90)
    plt.title(var.name+' partial autocorrelation')




def na_check(df):
    na = []
    cols = df.columns
    n = df.shape[0]
    
    for col in cols:
        na.append(1 - np.sum(pd.isnull(df[col]))/n)    
    na=pd.DataFrame({'value':na, 'name':cols})
    na.sort_values(by='value', ascending=True, inplace=True)
    na.reset_index(inplace=True, drop=True)

    plt.figure(figsize=(10,15))
    plt.barh(na.index, na['value'], align='center')
    plt.yticks(na.index, na['name'])
    
    plt.xlabel('Missing Value%')
    plt.title('Feature Missing Value Analysis')
    plt.tight_layout()
    plt.savefig('missing value check.jpg')
    
    return na
    
 
 
def value_count(df):
    
    count = []
    cols = df.columns
    
    for col in cols:
       count.append(len(df[col].value_counts()))
       
    count = pd.DataFrame({'value':count, 'name':cols})
    count.sort_values(by='value', ascending=True, inplace=True)
    count.reset_index(inplace=True, drop=True)

    return count
 
 
 
def parse_dummy(df,col):
    df[col].fillna(0, inplace=True)
    dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
    df = pd.concat([df,dummies], axis=1)
    del df[col]
    return df
    
    


    
 

def feature_plot(df,date_col,prediction_col):
    cols = list(df.columns)
    cols.remove(date_col)
    cols.remove(prediction_col)
    cols.remove('propertycountylandusecode')
    cols.remove('propertyzoningdesc')
    cols.remove('taxdelinquencyflag')
    for col in cols:
        
        plt.figure()
        plt.scatter(df[col], df[prediction_col])
        plt.xlabel(col)
        plt.ylabel(prediction_col)
        plt.savefig(col+' against '+prediction_col+'.jpg')
        

   
    
def data_partition(df,st,ed,spt,date_col,predition_col):
    
    ''' Partition data into training set and testing set
        
        Parameters
        ----------
        df: pandas dataframe
            data to be partitioned
        st: string
            start timestamp of the training set
        ed: string
            end timestamp of the testing set
        spt: string
            timestamp splitting training and testing set
        
        Returns
        -------
        training: pandas dataframe
        testing: pandas dataframe
    '''
    
    training = df.ix[(df[date_col]>st)&(df[date_col]<=spt),:]
    testing = df.ix[(df[date_col]>spt)&(df[date_col]<=ed)]
    train_y = training[predition_col]
    train_x = training.ix[:,training.columns != predition_col]
    test_y = testing[predition_col]
    test_x = testing.ix[:,testing.columns != predition_col]

    return train_x,train_y,test_x,test_y
    
    
    
    
def regression(trainx,trainy,testx,testy,cov='HAC',nw_maxlags=12,pred_alpha=0.125):
    
    if cov=='HAC':
        result=sm.OLS(trainy,trainx)\
        .fit(cov_type=cov,cov_kwds={'maxlags':nw_maxlags}) 
    else:
        result=sm.OLS(trainy,trainx)\
        .fit(cov_type=cov) 
   
    print(result.summary())
   
    prstd, iv_l, iv_u = wls_prediction_std(result,exog=testx,alpha=pred_alpha)
    y_hat=result.predict(testx)  
    pred=pd.DataFrame({'pred':y_hat,'ub':iv_u,'lb':iv_l},index=testx.index)
    
    return pred
