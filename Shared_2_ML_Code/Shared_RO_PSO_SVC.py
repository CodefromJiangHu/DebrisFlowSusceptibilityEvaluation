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
from ROC_Fig import acu_curve, save, cross_validation_CNN_plot_ROC, writeMessage, cross_validation_SVC_plot_ROC
import AIModelControl as aimCL
from CrossValidation_Plot_Fig import get_groups, plot_cross_validation_samples_distribution
from sklearn.model_selection import (TimeSeriesSplit, KFold, ShuffleSplit,
                                     StratifiedKFold, GroupShuffleSplit,
                                     GroupKFold, StratifiedShuffleSplit)
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from DataPostPrecessingControl import post_date_processing
import DataPreDrocessingControl as dpdCL
import ExcelControl as excelCL
import shap
import eli5
from eli5.sklearn import PermutationImportance
import matplotlib.pyplot as plt

def optimalizing_PSO_SVC(p):

    c, gamma = p
    cv = StratifiedShuffleSplit
    n_splits = 10
    this_cv = cv(n_splits=n_splits, random_state=6)
    param = {
        "C": [c],
        "kernel": ["rbf"],
        "probability": [True],
        "decision_function_shape": ["ovo"],
        "gamma": [gamma]}  
    model = aimCL.get_SVC_GridSearch()
    gc = GridSearchCV(model, param_grid=param, cv=this_cv) 
    gc.fit(X_train, Y_train.ravel())
    best_model = gc.best_estimator_
    Y_prob = best_model.predict(X_test)
    mse = mean_squared_error(Y_test, Y_prob)
    print('best_c:', c, '   ', 'best_gamma:', gamma)
    print('mean_squared_error: ', mse)
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
    modelNmae = "RO-PSO-SVC"
    flag_post = True
    flagSimle = False
    flagUseGridSearchCV = True
    flag_PSO = False
    flagExplainImportance_Shap = False
    flagExplainImportance_Eli5 = False
    flag_plot_crossvalidation = False

    n_splits = 10
    c_optimal, gamma_optimal = 64.6924, 0.0225
    str_para_vary = str(c_optimal) + "_" + str(gamma_optimal)
    # load data
    df_totalyangben_environment = pd.read_excel('inputs/Catchment_Details_Dadu_River.xlsx')
    df_yangben_environment = pd.read_excel('inputs/Yangben_Details_Dadu_River_RO_Output.xlsx')
    row_size = 1780
    colum_size = 8
    yangben_row_size = 558
    yangben_colum_size = 8
    list_fields_names = ["f_suscep", "f_dem", "f_ic", "f_soiltrength", "f_vegload", "Gully_J_Sd", "??????Sd", "??????Sd"]
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
    # (2)divide data
    X_train, X_test, Y_train, Y_test = dpdCL.dataprocessing_third_split(X_yangben_std, Y_yangben, 0.3, 0)

    if flag_plot_crossvalidation:

        cv = StratifiedShuffleSplit
        this_cv = cv(n_splits=n_splits, random_state=6)
        percentiles_classes = [0.5, 0.5]
        num_groups = 10
        groups = get_groups(len(X_train), num_groups)
        plot_cross_validation_samples_distribution(len(X_train), X_train, percentiles_classes, groups, n_splits,
                                                   this_cv)
        mode_classifier = aimCL.get_SVC_trainModel(c_optimal, gamma_optimal, "rbf", "ovo")
        cross_validation_SVC_plot_ROC(mode_classifier, X_train, Y_train, n_splits, cv, groups,
                                      str_para_vary, False)


    if flag_PSO:
        start_time = datetime.datetime.now()
        pso = PSO(func=optimalizing_PSO_SVC, n_dim=2, pop=40, max_iter=150, lb=[2.0e-8, 2.0e-8], ub=[100, 10.0], w=0.8, c1 = 0.5, c2 = 0.5)
        pso.record_mode = True
        pso.run()
        print('best_x is ', pso.gbest_x, 'best_y is', pso.gbest_y)

        end_time = datetime.datetime.now()
        # report time
        if flag_recordtime:
            time_dif = get_timedif_seconds(start_time, end_time)
            writeMessage("??????????????????????????????(s)???"+str(time_dif), "outputs/" + modelNmae + "_???????????????????????????.txt")
        c_optimal, gamma_optimal = pso.gbest_x[0], pso.gbest_x[1]
        # optimal parameters
        str_para_vary = str(pso.gbest_x) + "_" + str(pso.gbest_y)
        plt.plot(pso.gbest_y_hist)
        plt.show()

    if flag_evaluation:
        
        model = aimCL.get_Trained_SVC_GridSearch(X_train, Y_train, c_optimal, gamma_optimal, n_splits)
        if flag_post:
            post_date_processing(model, modelNmae, X_test, Y_test, X_yangben_std, Y_yangben, X_total_std, str_para_vary,
                                     flagSimle, df_totalyangben_environment)
        if flagExplainImportance_Shap:
            
            df_X_test = pd.DataFrame(X_test, columns=list_fields_names_plot)
            explainer = shap.KernelExplainer(model.predict_proba, X_test, link="logit")
            shap_values = explainer.shap_values(df_X_test, nsamples=100)
            shap_values = shap_values[1]
            print(shap_values.shape)
            shap.summary_plot(shap_values, df_X_test, plot_type="bar")
            shap.summary_plot(shap_values, df_X_test)
            # print importance
            abs_shap_values = np.abs(shap_values)
            result_importance = np.mean(abs_shap_values, axis=0)
            print(result_importance)
            df_tem = pd.DataFrame([result_importance], columns=list_fields_names_plot)
            df_tem.to_excel("outputs/" + str(str_para_vary) + modelNmae + "_Shap_Explain_importance.xlsx",
                            sheet_name="importance",
                            index=False, header=True)
        if flagExplainImportance_Eli5:
            perm = PermutationImportance(model).fit(X_test, Y_test.ravel())
            results_importance = [perm.feature_importances_, perm.feature_importances_std_]
            df_tem = pd.DataFrame(results_importance, columns=list_fields_names_plot)
            df_tem.to_excel("outputs/" + str(str_para_vary) + modelNmae + "_Eli5__Explain_importance.xlsx",
                            sheet_name="importance_and_std",
                            index=False, header=True)



