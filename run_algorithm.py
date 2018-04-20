# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 21:53:48 2018

Run this file to excute the preprocessing, training, predicting and also save/ load the model

@author: Jiang Ji
"""
from preprocessing import get_processed_data, train_valid_test_split
from train_models import train_all_models
from save_load import save_all_models_and_data, load_all_models_and_data
from second_layer import second_layer_optimizer
import numpy as np


# =============================================================================
# -------------------------------Prepare Data----------------------------------
# =============================================================================
# get processed data and split them into train, valid and test set
# le is the labelencoder which can be used to turn encoded label back to it's
# original form.
# Change the parameter n_classes to 5, if a 5 class classification is desired
X, y, le = get_processed_data(n_classes = 5)
(X_train, X_valid, X_test, y_train, y_valid, y_test) = train_valid_test_split(X, y, test_split = 0.15, valid_split = 0.2)



# =============================================================================
# -------------------------------First  layer----------------------------------
# =============================================================================
# =============================================================================
# excute the first layer of the model
# 2 model sets will be trained at this layer based on train and train+valid set
# =============================================================================


# train first modelset, only on train set
# best_parameters is the best parameters get from the models used Gridsearch
# or randomsearch, so they can be used in the second training
Models_train, best_parameters = train_all_models(X_train, y_train)



# train second modelset, on train + valid set
# best_parameters are used, so no need to tune those hype-parameter on this training
Models_train_valid = train_all_models(np.vstack((X_train, X_valid)), 
                                      np.vstack((y_train, y_valid)),
                                      best_parameters)



# =============================================================================
# -------------------------------Save/ load Models-----------------------------
# =============================================================================
# save the current model and data with it's train, valid, test split if the results
# are interesting
last_saved_folder = save_all_models_and_data(Models_train,
                                             Models_train_valid,
                                             (X_train, X_valid, X_test, y_train, y_valid, y_test),
                                             folder = 'new_run')

# A load_all_models_and_data function is provided to load saved model and data 
# with it's train, valid, test split, comment the save_all_models_and_data() 
# funciton above and uncomment the bellow line to do so

# Models_train, Models_train_valid, (X_train, X_valid, X_test, y_train, y_valid, y_test) = load_all_models_and_data('new_run')




# =============================================================================
# -------------------------------Sedond layer----------------------------------
# =============================================================================
# excute the second layer
# get the final prediction on test set and the weight
# assigned on each model used in the 1st layer
# =============================================================================
w_Opt, y_Opt = second_layer_optimizer(Models_train, 
                                      Models_train_valid, 
                                      (X_valid, X_test, y_valid, y_test),
                                      class_weights = [30, 1, 1, 1, 1])

