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
