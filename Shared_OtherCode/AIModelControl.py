# -*- coding: utf-8 -*-

# <editor-fold desc="common lib">
# common lib
import pandas as pd
import numpy as np
import math
from pandas import read_csv
import matplotlib.pyplot as plt

# model training
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
# SVC 
from sklearn.svm import SVC
from openpyxl import Workbook
from sklearn.metrics import roc_curve, auc  
from ROC_Fig import acu_curve
# Logistic 
from sklearn import linear_model, datasets
# XGBoost 
import xgboost as xgb
# RandomForest 
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score  
from sklearn import metrics
from xgboost.sklearn import XGBClassifier

from sklearn.model_selection import (TimeSeriesSplit, KFold, ShuffleSplit,
                                     StratifiedKFold, GroupShuffleSplit,
                                     GroupKFold, StratifiedShuffleSplit)
from CrossValidation_Plot_Fig import get_groups
from sklearn.model_selection import GridSearchCV
# </editor-fold>
# <editor-fold desc="LSTM">
# convert an array of values into a dataset matrix
def LSTM_create_dataset(dataset):  
    data = []
    for i in range(len(dataset)):
        a = dataset[i]
        data.append(a)  
    return np.array(data)  


def LSTM_getScaler(input_np_array):
    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))  
    dataset = scaler.fit_transform(input_np_array)
    return scaler, dataset


def LSTM_inverse_Scaler(input_scaler, standardedMatrix):
    m = standardedMatrix.shape[1]
    standarded_Matrix = standardedMatrix
    inverse_standarded_Matrix = input_scaler.inverse_transform(standarded_Matrix)
    inverse_total_Y = inverse_standarded_Matrix[:, m - 1:]
    return inverse_standarded_Matrix, inverse_total_Y


def LSTM_conver_into_input_form(X_matrix, steps):
    trainX = np.reshape(X_matrix, (X_matrix.shape[0], steps, X_matrix.shape[1]))
    return trainX


def LSTM_trainModel(X_train, Y_train, input_epochs, steps):
    row, colum = X_train.shape
    trainX = X_train  
    trainY = Y_train 
    # reshape input to be [samples, time steps, features] 
    trainX = np.reshape(trainX, (trainX.shape[0], steps, trainX.shape[1]))
    # build LSTM model
    model = Sequential()
    model.add(LSTM(11, input_length=steps, input_dim=colum))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='mean_squared_error', optimizer='adam')  
    model.fit(trainX, trainY, epochs=input_epochs, batch_size=1, verbose=2) 
    return model


def LSTM_readCSVAsArray(CSV_FilePath):
    # data = read_csv('data_multi.csv') 
    data = read_csv(CSV_FilePath)  
    dataset_ini = data.values;
    dataset_ini = np.array(dataset_ini.astype('float32')) 
    return dataset_ini

def LSTM_ROC_Parameters_Cal(Y_test, prepro):
    # ROC
    acu_curve(Y_test, prepro[:, 0])
    fpr, tpr, threshold = roc_curve(Y_test, prepro[:, 0])  
    roc_auc = auc(fpr, tpr)  
    TEMLIST = np.zeros(shape=[len(fpr), 3], dtype='float32')
    TEMLIST[:, 0] = fpr
    TEMLIST[:, 1] = tpr
    TEMLIST[:, 2] = threshold
    save(TEMLIST, "ROC_resuts.xlsx")
    return fpr, tpr, roc_auc
# </editor-fold>

# <editor-fold desc="SVC">
def SVC_trainModel(X_train, Y_train):
    # SVC
    svcmodel = SVC(C=0.8, kernel='linear', decision_function_shape='ovo', probability=True)
    svcmodel.fit(X_train, Y_train)
    return svcmodel
# <editor-fold desc="SVC">
def get_SVC_trainModel(_c,_gamma,_kernel,_decision_function_shape):
    # SVC
    # svcmodel = SVC(C=0.8, kernel='linear', decision_function_shape='ovo', probability=True)
    svcmodel = SVC(C=_c, kernel=_kernel, decision_function_shape=_decision_function_shape, probability=True, gamma=_gamma)
    return svcmodel
def get_SVC_GridSearch():
    # SVC
    # svcmodel = SVC(C=0.8, kernel='linear', decision_function_shape='ovo', probability=True)
    svcmodel = SVC(probability=True, kernel="rbf")
    return svcmodel

def get_Trained_SVC_GridSearch(X_train, Y_train, c, gamma, n_splits):
    
    cv = StratifiedShuffleSplit
    this_cv = cv(n_splits=n_splits, random_state=6)
    param = {
        "C": [c],
        "kernel": ["rbf"],
        "probability": [True],
        "decision_function_shape": ["ovo"],
        "gamma": [gamma]} 
    model = get_SVC_GridSearch()
    gc = GridSearchCV(model, param_grid=param, cv=this_cv)  
    gc.fit(X_train, Y_train.ravel())
    best_model = gc.best_estimator_
    return best_model

# </editor-fold>

# <editor-fold desc="RandomForest">
def randomForest_TrainModel(X_train, Y_train):
    model = RandomForestClassifier()
    model.fit(X_train, Y_train)
    print(model.feature_importances_)
    return model
def get_randomForest_trainModel(min_samples_split, _max_depth, _n_estimators,_min_samples_leaf):
    # criterion’ = ‘gini   max_depth’ = 18  n_estimators’ = 250  oob_score’ = True
    # model = RandomForestClassifier(criterion="gini", max_depth= 16, n_estimators=100, noob_score=True)
    # model = RandomForestClassifier(criterion=_criterion, max_depth=_max_depth, n_estimators=_n_estimators, noob_score=True)
    model = RandomForestClassifier(max_depth=_max_depth, n_estimators=_n_estimators,min_samples_leaf = _min_samples_leaf, min_samples_split = min_samples_split)
    return model
def get_randomForest_GridSearch():
    model = RandomForestClassifier()
    return model
def get_Trained_RF_GridSearch(X_train, Y_train,  n_estimators_optimal, max_depth_optimal, n_splits):
    cv = StratifiedShuffleSplit
    this_cv = cv(n_splits=n_splits, random_state=6)
    param = {
        "n_estimators": [n_estimators_optimal],
        "max_depth": [max_depth_optimal],
        "min_samples_split": [2],
        "min_samples_leaf": [15]}  
    # model
    model = get_randomForest_GridSearch()
    gc = GridSearchCV(model, param_grid=param, cv=this_cv)  
    gc.fit(X_train, Y_train.ravel())
    best_model = gc.best_estimator_
    return best_model

# </editor-fold>
# <editor-fold desc="LogisticRegression">
def logis_TrainModel(X_train, Y_train):
    logreg = linear_model.LogisticRegression(C=1e5)
    logreg.fit(X_train, Y_train)
    return logreg
def get_logis_trainModel(X_train, Y_train):
    logreg = linear_model.LogisticRegression(C=1e5)
    logreg.fit(X_train, Y_train)
    return logreg


# </editor-fold>

# <editor-fold desc="XGBoost">
def get_xgboost_GridSearch():
   
    model = XGBClassifier(objective='binary:logistic')
    return model
def get_Trained_XGB_GridSearch(X_train, Y_train,  n_estimators_optimal, max_depth_optimal, learning_rate_optimal, n_splits):
    
    cv = StratifiedShuffleSplit
    this_cv = cv(n_splits=n_splits, random_state=6)

    param = {
        "n_estimators": [n_estimators_optimal],
        "max_depth": [max_depth_optimal],
        "subsample": [0.8],
        "learning_rate": [learning_rate_optimal]}  

    model = get_xgboost_GridSearch()
    gc = GridSearchCV(model, param_grid=param, cv=this_cv) 
    gc.fit(X_train, Y_train.ravel())
    best_model = gc.best_estimator_
    return best_model
def xgboost_TrainModel(X_train, Y_train):
    params = {
        'booster': 'gbtree',
        'objective': 'binary:logistic',
        'num_class': 3,
        'gamma': 0.1,
        'max_depth': 6,
        'lambda': 2,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'min_child_weight': 3,
        'slient': 1,
        'eta': 0.1,
        'seed': 1000,
        'nthread': 4,
    }
    plst = params.items()
    dtrain = xgb.DMatrix(X_train, Y_train)
    num_rounds = 500
    model = xgb.train(list(plst), dtrain, num_rounds)
    return model

def xgboost_TrainModel_Classifier(X_train, Y_train):
    m_class = xgb.XGBClassifier(
        learning_rate=0.1,
        n_estimators=1000,
        max_depth=5,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        nthread=4,
        seed=27)
    m_class.fit(X_train, Y_train)
    return m_class

def get_xgboost_trainModel(_subsample, _max_depth, _n_estimators,_learning_rate):
    m_class = XGBClassifier(
        # learning_rate=0.1,
        # n_estimators=1000,
        # max_depth=5,
        learning_rate=_learning_rate,
        n_estimators=_n_estimators,
        max_depth=_max_depth,
        gamma=0.2,
        subsample=_subsample,
        colsample_bytree=1,
        objective='binary:logistic',
        nthread=4,
        # silent=0,
        silent=1,
        seed=6)
    return m_class


def xgboost_ROC_Parameters_Cal(Y_test, prepro):
    accuracy = metrics.roc_auc_score(Y_test, prepro)
    print("验证集预测精度为："+str(accuracy))
    acu_curve(Y_test, prepro)
    fpr, tpr, threshold = roc_curve(Y_test, prepro)  
    roc_auc = auc(fpr, tpr)  
    TEMLIST = np.zeros(shape=[len(fpr), 3], dtype='float32')
    TEMLIST[:, 0] = fpr
    TEMLIST[:, 1] = tpr
    TEMLIST[:, 2] = threshold
    save(TEMLIST, "XGBoost_ROC_resuts.xlsx")
    return fpr, tpr, roc_auc



# </editor-fold>

# <editor-fold desc="common post-processing method">

def save(data, path):
    wb = Workbook()
    ws = wb.active 
    [h, l] = data.shape  
    for i in range(h):
        row = []
        for j in range(l):
            row.append(data[i, j])
        ws.append(row)
    wb.save(path)

def ROC_Parameters_Cal(Y_test, prepro):

    acu_curve(Y_test, prepro)
    fpr, tpr, threshold = roc_curve(Y_test, prepro)  
    roc_auc = auc(fpr, tpr) 
    TEMLIST = np.zeros(shape=[len(fpr), 3], dtype='float32')
    TEMLIST[:, 0] = fpr
    TEMLIST[:, 1] = tpr
    TEMLIST[:, 2] = threshold
    save(TEMLIST, "ROC_resuts.xlsx")
    return fpr, tpr, roc_auc

# </editor-fold>
