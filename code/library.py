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


from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.grid_search import ParameterGrid



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
    train_x = training.ix[:, ~training.columns.isin([predition_col,\
    'transactiondate','parceid'])]
    test_y = testing[predition_col]
    test_x = testing.ix[:, ~testing.columns.isin([predition_col,\
    'transactiondate','parceid'])]

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
    
    
    
    
    
def gridsearch(model,param_grid,x_train,y_train,x_test,y_test):
    
    ''' perform grid search of optimal parameters
    
        Parameters
        ----------
        x_train: pandas dataframe
            all feature values in training set
        y_train: pandas series
            all target values in training set
        x_test: pandas dataframe
            all feature values in testing set
        y_test: pandas series
            all target values in testing set
        model: a prediction model function
            model must output mse
        param_grid: dictionary
            contains all possible values of each parameter        
        
        returns
        -------
        result: dictionary
            contains the set of parameter values 
            generating the smallest mse
            
    '''
    
    parms=list(ParameterGrid(param_grid))
    maes=[]
    stds=[]
    for parm in parms:
        print(parm)
        mae,std,pred=model(x_train,y_train,x_test,y_test,**parm)    
        maes.append(mae)
        stds.append(std)
    result=pd.DataFrame({'parms':parms,'mae':maes,'std':stds})
    result.sort_values(by='mse',ascending=True,inplace=True)
    return result
     
    
     
    
    
    
def rf(x_train,y_train,x_test,y_test,n_trees,max_depth,\
max_features,plot_impt=False,**kwargs):
    
    ''' random forest model
    
        Parameters
        ----------
        x_train: pandas dataframe
            all feature values in training set
        y_train: pandas series
            all target values in training set
        x_test: pandas dataframe
            all feature values in testing set
        y_test: pandas series
            all target values in testing set
        n_trees: integer
            number of trees
        max_depth: integer
            maximum depth of each tree
        max_features: integer
            maximum number of features to be 
            considered when decising split point
        plot_impt: boolean (optional)
            plot feature importance or not
        **kwargs: other parameters 
            passed to RandomForestRegressor function
        
        returns
        -------
        mse: float
            mean squared error of predictions
        std: float
            standar error of prediction residuals
        rf_pred: pandas series
            predicted value   
            
    '''
    
    cols=x_train.columns
    rf=RandomForestRegressor(n_estimators=n_trees,\
    max_depth=max_depth,max_features=max_features,criterion='mae',**kwargs)
    rf.fit(x_train,y_train)
    feature_importance=rf.feature_importances_
    feature_importance=pd.DataFrame({'col':cols,'score':feature_importance})
    feature_importance.sort_values(by='score',ascending=False,inplace=True)
    rf_pred=rf.predict(x_test)
    mae=np.mean(np.abs(rf_pred-y_test))
    std=np.std(rf_pred-y_test)  
    
    if (plot_impt==True):
       fig, ax = plt.subplots() 
       ax.barh(np.arange(feature_importance.shape[0]), \
       feature_importance['score'], align='center',\
       color='green')
       ax.set_yticks(np.arange(feature_importance.shape[0]))
       ax.set_yticklabels(feature_importance['col'])
       ax.invert_yaxis()  # labels read top-to-bottom
       ax.set_xlabel('Feature Importance')
       ax.set_title('Random Forest Feature Importance')

    return mae,std,rf_pred
