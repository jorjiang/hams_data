# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 02:37:04 2018
This file contains functions to train the models used in the 1st layer, including:
    train_keras_model()
    train_gradient_boost()
    train_RF()
    train_LR()
    train_XGB
    
    train_all_models(): return a dictionaray of all trained models and the best_parameters
                        of the model that trained with grid/ random search
@author: Jiang Ji
"""
import numpy as np
import gc

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from xgboost.sklearn import XGBClassifier
import lightgbm as lgb

from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam

random_state=1
    
def train_keras_model(X_train, y_train, class_weight):
    """
    Function to train a Deep neural network.
    
    Parameters:
    ----------
    X_train: array-like, feature set
    y_train: array-like, labels
    class_weight: list
    
    Return: model
    ------
    """
    

    print('Training Deep neural network:') 
    print('------------------------------------------------------------')
    n_classes = y_train.shape[1]
    input_dim = X_train.shape[1]
    # create model
    model = Sequential()
    
    model.add(Dense(input_dim = input_dim, units = 80, activation='relu'))
    model.add(BatchNormalization(axis = 1))
    model.add(Dropout(0.5))
    model.add(Dense(units = 30, activation='relu'))
    model.add(BatchNormalization(axis = 1))
    model.add(Dropout(0.5))
    model.add(Dense(n_classes, activation='softmax'))
    
    # Set earlystop
    earlystop = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=10, \
                          verbose=1, mode='auto')    
    # Compile model
   
    model.compile(loss = 'categorical_crossentropy', optimizer=Adam(lr = 0.001), metrics = ['accuracy'])
    model.fit(X_train, y_train,
              validation_split=0.20,
              batch_size=200,
              epochs = 100000,
              shuffle = True,
              callbacks = [earlystop],
              class_weight=class_weight) 
    gc.collect();
    return model




def train_gradient_boost(X_train, y_train):
    """
    Function to train gradient boost model with lightgbm.
    
    Parameters:
    ----------
    X_train: array-like, feature set
    y_train: array-like, labels
    
    Return: model
    ------
    """    
    n_classes = y_train.shape[1]
    print('Training Gradient boost model with lightgbm:')
    print('------------------------------------------------------------')
    
    # aggregate y_train from 1-Hot to sigle column, that's just how lightgbm works with
    y_train = y_train.argmax(1).squeeze()
    
    # Create validation set for early stop
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)
    
    # differnt parameters for binary and multiclass classification
    if n_classes == 2:
        objective = 'binary'
        metric = 'xentropy'
        n = 1
    elif n_classes == 5:
        objective = 'multiclassova'
        metric = 'multi_logloss'
        n = 5
        
    # set parameters        
    params = {
        'boosting_type': 'gbdt',
        'objective': objective,
        'num_class': n,
        'metric': metric,
        'max_depth': 7,
        'num_leaves': 31,
        'learning_rate': 0.03,
        'feature_fraction': 0.85,
        'bagging_fraction': 0.85,
        'bagging_freq': 5,
        'num_threads': 4,
        'lambda_l2': 1,
        'min_gain_to_split': 0,
        'is_unbalance': True ,
        'class_weight': 'balanced'
    }  
    
    # Train model    
    model = lgb.train(
        params,
        lgb.Dataset(X_train, y_train),
        num_boost_round=10000,
        valid_sets=[lgb.Dataset(X_val, y_val)],
        early_stopping_rounds=300,
        verbose_eval=100,
    )
    gc.collect();
    return model


def train_RF(X_train, y_train, best_parameters = None):
    """
    Function to train Random forest model, optionally with random search to find
    the best hyper parameters.
    
    Parameters:
    ----------
    X_train: array-like, feature set
    y_train: array-like, labels
    best_parameters: if None, no best_parameters is given, model will be trained
                     with random search with K-foldCV.
                     if best_parameters is given, no search will be excuted, the
                     model will be trained with the given hyper parameters
                  
    Return: model, best_parameters
    ------
    """    
    if best_parameters == None:
        RF = RandomForestClassifier()

        # If best_parameters is not given, excute Randomzied search to find the
        #best_parameters
        print('Training Random forest with RandomizedSearchCV:')
        print('------------------------------------------------------------')
        
        from sklearn.model_selection import RandomizedSearchCV
    
        # Create the random grid
        random_grid = {'n_estimators': [100, 200, 300, 1000],
                       'max_depth': [6, 7, 8, 9],
                       'min_samples_split': [50, 100, 200],
                       'min_samples_leaf': [10, 20, 50],
                       }
        
        # Use the random grid to search for best hyperparameters
        # Random search of parameters, using 'cv' fold cross validation, 
        # search across n_iter different combinations, and use all available cores
        RF_random = RandomizedSearchCV(estimator = RF,
                                       param_distributions = random_grid,
                                       n_iter = 25,
                                       cv = 3,
                                       verbose= 10,
                                       random_state=random_state, n_jobs = -1)
        
        # Fit the random search model
        model = RF_random.fit(X_train, y_train.argmax(1).squeeze())
        gc.collect();
        best_parameters = model.best_params_
        
    else:
        # If best_parameter is given, then train with best parameters instead
        # of doing randomized search
        print('Training Random forest with best parameters from last randomized search:')
        print('------------------------------------------------------------')
        best_parameters['n_jobs'] = -1
        model = RandomForestClassifier(**best_parameters).fit(X_train, y_train.argmax(1).squeeze())
    gc.collect(); 
    return model, best_parameters
    


def train_LR(X_train, y_train):
    """
    Function to train logistic regression model with grid search.
    
    Parameters:
    ----------
    X_train: array-like, feature set
    y_train: array-like, labels
    
    Return: model
    ----
    """
    
    from sklearn.model_selection import GridSearchCV
    print('Training LR with Grid Search:')
    print('------------------------------------------------------------')
    # Create the random grid
    grid = {'penalty': ['l1', 'l2'],
            'C': [0.5, 1, 2],
           }

    LR = LogisticRegression(random_state=random_state)
    
    # Train model with grid search with K-foldCV
    LR_random = GridSearchCV(estimator = LR,
                             param_grid = grid,
                             cv = 3,
                             verbose= 10,
                             n_jobs = -1)
    model = LR_random.fit(X_train, y_train.argmax(1).squeeze())
    gc.collect();    
    return model



def train_XGB(X_train, y_train):
    """
    Function to train gradient boost model with xgboost.
    
    Parameters:
    ----------
    X_train: array-like, feature set
    y_train: array-like, labels
    
    Return: model
    ----
    """    
    print('Training XGB:')
    print('------------------------------------------------------------')
    
    n_classes = y_train.shape[1]
    # Create validation set for early stop
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)
    
    
    
    
    # differnt parameters for binary and multiclass classification
    if n_classes == 2:
        objective = 'binary:logistic'
        eval_metric = 'logloss'
    elif n_classes == 5:
        objective = 'multi:softmax'
        eval_metric = 'mlogloss'
    # set hyper parameters        
    XGB = XGBClassifier(max_depth=7,
                        learning_rate=0.03,
                        objective = objective,
                        n_estimators=1000,
                        subsample=0.5,
                        colsample_bytree=0.5,
                        seed=1)
    
    # train model
    model = XGB.fit(X_train, y_train.argmax(1).squeeze(),
                  early_stopping_rounds=150,
                  eval_metric=eval_metric,
                  eval_set=[(X_val, y_val.argmax(1).squeeze())],
                  verbose=100)
    gc.collect();
    return model



def train_all_models(X_train, y_train, best_parameters = None):
    """
    Function to train All models and save the trained modles to a dictionary object,
    the models trained with grid/ random search will also return it's best parameters
    been found.
    
    Parameters:
    ----------
    X_train: array-like, feature set
    y_train: array-like, labels
    best_parameters: dictionary of parameters
    
    Return: 
    ----
    Models: A dictionary of all trained models and a dictinary of tuned hyper parameters
    
    """   
    
    
    n_classes = y_train.shape[1]
    LR = train_LR(X_train, y_train)    
    XGB = train_XGB(X_train, y_train)
    Dnn = train_keras_model(X_train, y_train, class_weight = np.ones((n_classes,1)))
    Lgbm = train_gradient_boost(X_train, y_train)
    RF, parameters = train_RF(X_train, y_train, best_parameters)


    
    Models = {
        'Dnn'  : Dnn, 
        'Lgbm' : Lgbm, 
        'LR'  : LR, 
        'RF'  : RF, 
        'XGB' : XGB
        }
    
    gc.collect();
    if best_parameters == None:    
        return Models, parameters
    else:
        return Models