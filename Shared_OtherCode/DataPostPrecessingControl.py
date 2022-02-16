import numpy as np
from sklearn.metrics import roc_curve, auc, mean_squared_error 
from ROC_Fig import acu_curve, save, cross_validation_SVC_plot_ROC, writeMessage
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score
import math

def post_date_processing(_model, modelNmae, X_test, Y_test, X_yangben_std, Y_yangben, X_total_std, str_para_vary, flagSimle,df_totalyangben_environment):
    model = _model
    prepro = model.predict(X_test)
    loss = log_loss(Y_test, prepro)
    acc = accuracy_score(Y_test, prepro)
    print("acc:" + str(acc))
    writeMessage(str_para_vary + "," + str(loss) + "," + str(acc), "outputs/" + modelNmae + "_optimal_parameters.txt")
    prepro_yangben = model.predict_proba(X_test)
    prepro_yangben = np.array(prepro_yangben)
    fpr, tpr, threshold = roc_curve(Y_test, prepro_yangben[:, 1]) 
    roc_auc = auc(fpr, tpr) 
    # output auc value in to txt
    writeMessage("AUC：" + str(roc_auc), "outputs/" + modelNmae + "_optimal_parameters.txt")

    # calculate mse and rmse
    mse = mean_squared_error(Y_test, prepro_yangben[:, 1])
    rmse = math.sqrt(mse)
    writeMessage("MSE：" + str(mse), "outputs/" + modelNmae + "_optimal_parameters.txt")
    writeMessage("RMSE：" + str(rmse), "outputs/" + modelNmae + "_optimal_parameters.txt")


    TEMLIST = np.array((fpr, tpr), dtype='float32')
    if not (flagSimle):
        save(TEMLIST, "outputs/" + str(str_para_vary) + modelNmae + "_yangben_ROC_resuts.xlsx")

    # start_time = datetime.datetime.now()
    result_pro = model.predict_proba(X_total_std)
    result = model.predict(X_total_std)
    # end_time = datetime.datetime.now()
    # time_dif = get_timedif_seconds(start_time, end_time)
    result_pro_1 = np.array(result_pro[:, 1])
    # save results 
    df_totalyangben_environment['result_pro'] = result_pro_1
    df_totalyangben_environment['result'] = result
    if not (flagSimle):
        df_totalyangben_environment.to_excel("outputs/" + str(str_para_vary) + modelNmae + "sus_results.xlsx",
                                             sheet_name="Resutls",
                                             index=False, header=True)
    print("finish！")