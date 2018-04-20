# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 00:12:22 2018

This file contains functions pre-process the data, including:
    get_processed_data()
    train_valid_test_split()
    
@author: Jiang Ji
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import scale, LabelBinarizer
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split


def get_processed_data(n_classes = 5):
    """
    Function to read and pre-process dataset.
    
    Parameters:
    ----------
    n_classes: if = 5, data will be prepared for multi-class classification,
               if = 2, data will be prepared for binary classification
    
    Return: 
    ------
    X: Featuers
    y: labels
    le: sklearn encoder object, can be used to transfer encoded classes to its
        original form.
        
    """    
    #read data
    df = pd.read_csv('sample.csv', header=None)
    df.columns = map(str, df.columns)
       
    #remove highly related columns
    high_corr_col = np.arange(66, 76).tolist() + np.arange(169, 179).tolist()
    col_drop = [str(col) for col in high_corr_col]
    df = df.drop(col_drop, 1)    
    
    #separate features and labels
    X = df.drop(['295'], 1)
    y = df['295'] 
    
    X = scale(X) #normalize features X
    
    if n_classes == 2:
        aggregate_small_classes = {'A':'F',
                                   'B':'F',
                                   'C':'C',
                                   'D':'F',
                                   'E':'F'}
        y = y.map(aggregate_small_classes)
        #encode labels   
        le = LabelBinarizer().fit(y)
        y = le.transform(y)
        y = to_categorical(y)
    elif n_classes == 5:
        le = LabelBinarizer().fit(y)
        y = le.transform(y)
    else:
        print('Error, n_classes can only be 5 or 2')
    
    return X, y, le


def train_valid_test_split(X, y, test_split = 0.15, valid_split = 0.2):
    """
    Function to split data in to train, validation, test set in one go.
    
    Parameters:
    ----------
    X: Featuers
    y: labels
    test_split: default 0.15
    valid_split: default 0.2
    
    Return: 
    ------
    data: Tuple,
          in the form: (X_train, X_valid, X_test, y_train, y_valid, y_test)       
    """  
    X, X_test, y, y_test = train_test_split(X, y, test_size=.2)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=.25)
    data = (X_train, X_valid, X_test, y_train, y_valid, y_test)
    return data
    



# Check coloniality
# =============================================================================
# corr = df.corr()
# high_cor_list_1 = np.arange(66, 76).tolist()
# high_cor_list_2 = np.arange(169, 179).tolist()
# high_cor_list_1_2 = np.arange(76, 170).tolist()
# high_cor_list_2_2 = np.arange(180, 268).tolist()
# 
# def find_cor():
#     high_corr_list = {}
#     for col_1 in high_cor_list_1:
#         high_corr_list[str(col_1)] = pd.DataFrame(df[str(col_1)])
#         for col_2 in high_cor_list_1_2:
#             if corr.values[col_1, col_2] > 0.05:
#                 high_corr_list[str(col_1)] = pd.concat([high_corr_list[str(col_1)], df[str(col_2)]], 1)
#     for col_1 in high_cor_list_2:
#         high_corr_list[str(col_1)] = pd.DataFrame(df[str(col_1)])
#         for col_2 in high_cor_list_2_2:
#             if corr.values[col_1, col_2] > 0.05:
#                 high_corr_list[str(col_1)] = pd.concat([high_corr_list[str(col_1)], df[str(col_2)]], 1)
#     return high_corr_list
#                 
# xxx = find_cor()        
# xxxxx = xxx
# yes = []
# for key in xxx:
#     xxx[key] = xxx[key][xxx[key].sum(1) != 0]
#     yes.append((xxx[key].iloc[:,1:].sum(1) ^ xxx[key].iloc[:,0]).sum()/len(df))
# =============================================================================
