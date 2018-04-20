# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 17:37:54 2018
@author: Jiang Ji

This file contains Functions to save or load trained models along with it's corresponding
train, validation, test set, including:
    save_all_models_and_data()
    load_all_models_and_data()
    dump_model()
    load_pickled_model()

"""
import pickle
import pandas as pd
from keras.models import load_model
import datetime
import os
import gc


def save_all_models_and_data(Models_train, Models_train_valid, data, folder = None):
    
    """
    Function to save all the models and the corresponding train-valid-test splits.
    
    Parameters:
    ----------
    Models_train: first model sets to be saved
    Models_train_valid: second model sets to be saved
    data: the data to save, a Tuple in following format:
        (X_train, X_valid, X_test, y_train, y_valid, y_test)
    folder: the folder in which all models and data will be saved in, if = None,
        a new folder will be name as the current date time will be created, and 
        the models and data will be saved there
        
    Return:
    ------
    save_path: path everything is saved in
    """
    if folder == None:
        # create a unique folder to save everything if folder is not given
        save_path = './models/{}/'.format(str(datetime.datetime.now()).split('.')[0]).replace(':', '-').replace(' ', '-')
    else: 
        save_path = './' + folder
        
    # create folder if not exist
    if not os.path.exists(save_path):
        os.makedirs(save_path + '/Models_train')
        os.makedirs(save_path + '/Models_train_valid')
        os.makedirs(save_path + '/data')
        
    # save models
    # keras model are in a differnt format so it will be treated differently
    for nm, model in Models_train.items():
        if nm != 'Dnn':
            dump_model(save_path + '/Models_train/', model, nm)
        else:
            model.save(save_path + '/Models_train/' + 'Dnn.h5')
            
    for nm, model in Models_train_valid.items():
        if nm != 'Dnn':
            dump_model(save_path + '/Models_train_valid/', model, nm)
        else:
            model.save(save_path + '/Models_train_valid/' + 'Dnn.h5')
                   
    # save data            
    pd.to_pickle(data, save_path + '/data/' + 'data.pickle' )
    
    print('All Models and data saved!')
    gc.collect();
    return save_path;

        
def load_all_models_and_data(load_path):
    """
    Function to load all the models and the corresponding train-valid-test 
    splits from last run.
    
    Parameters: load_path
    ----------
            
    Return:
    ------
    Models_train, Models_train_valid: all models in this folder
    data: all data with train-valid-test splits in this folder
    """
    # load modles
    def load_models(models_folder):
        path = load_path + '/' +models_folder
        Models = {
            'Dnn'  : load_model (path + '/'+'Dnn.h5'), 
            'Lgbm' : load_pickled_model(path + '/'+ 'Lgbm.pickle'), 
            'LR'   : load_pickled_model(path + '/'+ 'LR.pickle'), 
            'RF'   : load_pickled_model(path + '/'+ 'RF.pickle'), 
            'XGB'  : load_pickled_model(path + '/'+ 'XGB.pickle')
            } 
        return Models
    
    Models_train = load_models('Models_train')
    Models_train_valid = load_models('Models_train_valid')
    
    # load data
    data = pd.read_pickle(load_path + '/data/data.pickle')
    
    print('All Models and data loaded!')
    gc.collect();
    return Models_train, Models_train_valid, data
        

    

def dump_model(path, model, model_name):
    """
    Function to dump a model to a pickle file with name model_name in 'path'.
    
    Parameters: path, model, model_name 
    ----------
   
    Return: None
    ------
    
    """
    with open(path + model_name +'.pickle', 'wb') as f:
        pickle.dump(model, f)
        
        
def load_pickled_model(path):
    """
    Function to load a model from a pickle file in 'path'.
    
    Parameters: path
    ----------
        
    Return: None
    ------
    
    """
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model