# -*- coding: utf-8 -*-
import numpy as np
from sko.GA import GA
from sko.PSO import PSO
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow import keras
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from ROC_Fig import acu_curve, save, cross_validation_CNN_plot_ROC, writeMessage
import AIModelControl as aimCL
from CrossValidation_Plot_Fig import get_groups
from sklearn.model_selection import (TimeSeriesSplit, KFold, ShuffleSplit,
                                     StratifiedKFold, GroupShuffleSplit,
                                     GroupKFold, StratifiedShuffleSplit)
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from DataPostPrecessingControl import post_date_processing
import DataPreDrocessingControl as dpdCL
import ExcelControl as excelCL

def optimalizing_PSO_RF(p):

    n_estimators, max_depth = p
    cv = StratifiedShuffleSplit
    n_splits = 10
    this_cv = cv(n_splits=n_splits, random_state=6)
    param = {
        "n_estimators": [int(n_estimators)],
        "max_depth": [int(max_depth)],
         "min_samples_split": [2],
        "min_samples_leaf": [15]}  
    # models
    model = aimCL.get_randomForest_GridSearch()
    gc = GridSearchCV(model, param_grid=param, cv=this_cv)  
    gc.fit(X_train, Y_train.ravel())
    best_model = gc.best_estimator_
    Y_prob = best_model.predict(X_test)
    mse = mean_squared_error(Y_test, Y_prob)
    print('best_x:', int(n_estimators), '   ', 'best_y:', int(max_depth), 'mean_squared_error: ', mse)
    # print('mean_squared_error: ', mse)
    return mse
def get_data_from_excel(df_excel, row_size,colum_size,list_fields_names,list_label,flag_traindata):
    X_total = np.zeros(shape=[row_size, colum_size], dtype='float32')
    Y_total = np.zeros(shape=[row_size, 1], dtype="bool")
    X_total = excelCL.convertDFtoArray(df_excel, list_fields_names)
    if flag_traindata:
        Y_total = excelCL.convertDFtoArray(df_excel, list_label)
    return X_total, Y_total
import datetime
from DateControl import get_timedif_seconds

if __name__ == '__main__':
    # parameters
    flag_recordtime = True
    flag_evaluation = True
    modelNmae = "PSO-RF"
    flagSimle = False
    flagUseGridSearchCV = True
    flagExplainImportance_Shap = False
    flagExplainImportance_Eli5 = True
    flag_PSO = False
    n_splits = 10
    n_estimators_optimal, max_depth_optimal = 50, 2
    str_para_vary = str(n_estimators_optimal) + "_" + str(max_depth_optimal)
    # load data
    df_totalyangben_environment = pd.read_excel('inputs/Catchment_Details_Dadu_River.xlsx')
    df_yangben_environment = pd.read_excel('inputs/Yangben_Details_Dadu_River.xlsx')
    row_size = 1780
    colum_size = 8
    yangben_row_size = 562
    yangben_colum_size = 8
    list_fields_names = ["f_suscep", "f_dem", "f_ic", "f_soiltrength", "f_vegload", "Gully_J_Sd", "流速Sd", "流深Sd"]
    list_fields_names_plot = ["PPI", "DEM", "IC", "ST", "VWL", "CD", "RV", "FD"]
    list_label = ["label"]
    X_total, Y_total = get_data_from_excel(df_totalyangben_environment, row_size, colum_size, list_fields_names,
                                           list_label, False)
    X_yangben, Y_yangben = get_data_from_excel(df_yangben_environment, yangben_row_size, yangben_colum_size,
                                               list_fields_names, list_label, True)
    # (1)data standardlization
    sc = dpdCL.dataprocessing_first_getStandardScaler(X_total)
    X_total_std = dpdCL.dataprocessing_second_StandardScaler_transform(sc, X_total)
    X_yangben_std = dpdCL.dataprocessing_second_StandardScaler_transform(sc, X_yangben)
    # (2)divide samples
    X_train, X_test, Y_train, Y_test = dpdCL.dataprocessing_third_split(X_yangben_std, Y_yangben, 0.3, 0)

    if flag_PSO:
        start_time = datetime.datetime.now()
        pso = PSO(func=optimalizing_PSO_RF, n_dim=2, pop=20, max_iter=150, lb=[30, 2], ub=[1000, 30], w=0.8, c1 = 0.5, c2 = 0.5)
        pso.record_mode = True
        pso.run()
        print('best_x is ', pso.gbest_x, 'best_y is', pso.gbest_y)
        end_time = datetime.datetime.now()
        # report time
        if flag_recordtime:
            time_dif = get_timedif_seconds(start_time, end_time)
            writeMessage("最优参数调试时间耗时(s)："+str(time_dif), "outputs/" + modelNmae + "_优化器参数调试结果.txt")
        n_estimators_optimal, max_depth_optimal = int(pso.gbest_x[0]), int(pso.gbest_x[1])
        # optimal parameters
        writeMessage("最优参数为：" + str(n_estimators_optimal)+"_"+str(max_depth_optimal), "outputs/" + modelNmae + "_优化器参数调试结果.txt")
        str_para_vary = str(n_estimators_optimal) + "_" + str(max_depth_optimal)
        plt.plot(pso.gbest_y_hist)
        plt.show()

    if flag_evaluation:
        # obtain model
        model = aimCL.get_Trained_RF_GridSearch(X_train, Y_train, n_estimators_optimal, max_depth_optimal, n_splits)
        post_date_processing(model, modelNmae, X_test, Y_test, X_yangben_std, Y_yangben, X_total_std, str_para_vary,
                                 flagSimle, df_totalyangben_environment)


