# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 00:12:22 2018

This file contains fuctions and Classes used for the second layer optimization, including:
    
    objf_ens():               objective function to minimize
    second_layer_optimizer(): the fucntion excute the second layer optimization
                              return and disply the results
    
    Class Opt:                optimizer built based on sklearn BaseEstimator
    
@author: Jiang Ji
"""

import numpy as  np
from sklearn.metrics import log_loss, confusion_matrix, accuracy_score
from sklearn.base import BaseEstimator
from scipy.optimize import minimize
from tabulate import tabulate

def objf_ens(w, Xs, y, n_class=2):
    """
    Function to be minimized in the ensembler.
    
    Parameters:
    ----------
    w: array-like, shape=(n_preds)
       Candidate solution to the optimization problem (vector of weights).
    Xs: list of predictions to combine
       Each prediction is the solution of an individual classifier and has a
       shape=(n_samples, n_classes).
    y: array-like sahpe=(n_samples,)
       Class labels
    n_class: int
       Number of classes in the problem
    
    Return:
    ------
    score: Score of the candidate solution.
    """
    #Constraining the weights for each class to sum up to 1.
    #This constraint can be defined in the scipy.minimize function, but doing 
    #it here gives more flexibility to the scipy.minimize function 
    #(e.g. more solvers are allowed).
    n_class = y.shape[1]
    
    w_range = np.arange(len(w))%n_class 
    for i in range(n_class): 
        w[w_range==i] = w[w_range==i] / np.sum(w[w_range==i])
        
    sol = np.zeros(Xs[0].shape)
    for i in range(len(w)):
        sol[:, i % n_class] += Xs[int(i / n_class)][:, i % n_class] * w[i] 
        
    #Using log-loss as objective function (different objective functions can be used here). 
    score = log_loss(y, sol)   
    return score
    

class Opt(BaseEstimator):
    """
    Given a set of predictions $X_1, X_2, ..., X_n$, where each $X_i$ has
    $m=12$ clases, i.e. $X_i = X_{i1}, X_{i2},...,X_{im}$. The algorithm finds the optimal 
    set of weights $w_{11}, w_{12}, ..., w_{nm}$; such that minimizes 
    $log\_loss(y_T, y_E)$, where $y_E = X_{11}*w_{11} +... + X_{21}*w_{21} + ... 
    + X_{nm}*w_{nm}$ and and $y_T$ is the true solution.
    """
    def __init__(self, n_class=2):
        super(Opt, self).__init__()
        self.n_class = n_class
        
    def fit(self, X, y):
        """
        Learn the optimal weights by solving an optimization problem.
        
        Parameters:
        ----------
        Xs: list of predictions to be ensembled
           Each prediction is the solution of an individual classifier and has 
           shape=(n_samples, n_classes).
        y: array-like
           Class labels
        """
        
        Xs = np.hsplit(X, X.shape[1]/self.n_class)
        #Initial solution has equal weight for all individual predictions.
        x0 = np.ones(self.n_class * len(Xs)) / float(len(Xs)) 
        #Weights must be bounded in [0, 1]
        bounds = [(0,1)]*len(x0)   
        #Calling the solver (constraints are directly defined in the objective
        #function)
        res = minimize(objf_ens, x0, args=(Xs, y, self.n_class), 
                       method='L-BFGS-B', 
                       bounds=bounds, 
                       )
        self.w = res.x
        self.classes_ = np.unique(y)

        return self
    
    def predict_proba(self, X):
        """
        Use the weights learned in training to predict class probabilities.
        
        Parameters:
        ----------
        Xs: list of predictions to be ensembled
            Each prediction is the solution of an individual classifier and has 
            shape=(n_samples, n_classes).
            
        Return:
        ------
        y_pred: array_like, shape=(n_samples, n_class)
                The ensembled prediction.
        """
        Xs = np.hsplit(X, X.shape[1]/self.n_class)
        y_pred = np.zeros(Xs[0].shape)
        for i in range(len(self.w)):
            y_pred[:, i % self.n_class] += \
                   Xs[int(i / self.n_class)][:, i % self.n_class] * self.w[i]
           
        return y_pred     
    
def second_layer_optimizer(Models_train,
                           Models_train_valid,
                           valid_test_data,
                           class_weights = None,
                           cf_matrix = True):
    """
    Function to run the second-layer optimizer based on the output of frist
    layer on validation set.
    
    Parameters:
    ----------
    Models: Dictionary, 
       contains all the models have been trained in the first layer.
       
    valid_test_data: Tuple,
       contains validation set and test set, in the form of
       (X_valid, X_test, y_valid, y_test).
    
    class_weights: list,  
       
    cf_matrix: Boolean,
        whether or not print confusion_matrix, default = True
        
    Return:
    ------
    w_Opt: N x M array
            optimized weight from 2nd layer optimizer, where:
            N is the number of models used in the first layer,
            M is the number of classes
    
    y_Opt: final predicted probability from 2nd layer optimizer with the weight
            w_Opt
    """

    X_valid, X_test, y_valid, y_test = valid_test_data
    if class_weights == None:
        # if class_weights are not given, it will be equeal for every class
        class_weights == np.ones(y_valid.shape[1])    
    n_classes = y_valid.shape[1] #get class number
    p_valid = [] #a list to save predictions of each 1st-layer-model on validation set
    p_test = [] #a list to save predictions of each 1st-layer-model on test set
    print('Performance of individual classifiers (1st layer) on X_test')   
    print('------------------------------------------------------------')
    
    for nm, model in Models_train.items():
        if nm == 'Lgbm':
        #Use 'if statements' here because lightgbm use predict() to predict probabilities while others use predict_proba()
            #First run. Training on (X_train, y_train) and predicting on X_valid.
            yv = model.predict(X_valid) 
            #Second run. Training on (X, y) and predicting on X_test.p
            yt = Models_train_valid[nm].predict(X_test)
            if n_classes == 2:
                yv = np.vstack((1-yv, yv)).T
                yt = np.vstack((1-yt, yt)).T

        else:
            #First run. Training on (X_train, y_train) and predicting on X_valid.
            yv = model.predict_proba(X_valid) 
            #Second run. Training on (X, y) and predicting on X_test.
            yt = Models_train_valid[nm].predict_proba(X_test)      
        p_valid.append(yv)
        p_test.append(yt)
        
        
        #Printing out the performance of the classifier        

        print('{:10s} {:2s} {:1.7f}'.format('%s: ' %(nm), 'logloss  =>', log_loss(y_test, yt)))
    print('')
    
    print('Performance of optimization based ensemblers (2nd layer) on X_test')   
    print('------------------------------------------------------------')
        
    #Creating the data for the 2nd layer.
    XV = np.hstack(p_valid)
    XT = np.hstack(p_test)  
            
    #EN_optB
    opt = Opt(n_classes)
    opt.fit(XV, y_valid)
    w_Opt = opt.w
    y_Opt = opt.predict_proba(XT)
    print('{:2s} {:1.7f}'.format('Final logloss  =>', log_loss(y_test, y_Opt)))
        
    print('Weights for each classifer:class')
    
    print('|--------------------|')
    wB = np.round(w_Opt.reshape((-1,n_classes)), decimals=2)
    wB = np.hstack((np.array(list(Models_train.keys()), dtype=str).reshape(-1,1), wB))
    print(tabulate(wB, headers=['y%s'%(i) for i in range(n_classes)], tablefmt="orgtbl"))
    
    y_pred = y_Opt * class_weights
    y_label_test = y_test.argmax(1).squeeze()
    y_label_pred = y_pred.argmax(1).squeeze()
    
    if cf_matrix: 
        print('------------------------------------------------------------')
        print('Confusion Matrix:\n', confusion_matrix(y_label_test, y_label_pred))
    
    print('------------------------------------------------------------')
    print('accuracy:', accuracy_score(y_label_test, y_label_pred))
    return w_Opt, y_Opt    