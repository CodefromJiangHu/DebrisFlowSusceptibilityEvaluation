
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
def dataprocessing_simple(X_yangben, Y_taget, rate_split):
    sc = StandardScaler()
    sc.fit(X_yangben)
    X_yangben_std = sc.transform(X_yangben)
    X_train, X_test, Y_train, Y_test = train_test_split(X_yangben_std, Y_taget, test_size=rate_split, random_state=0)
    return X_train, X_test, Y_train, Y_test

def dataprocessing_first_getStandardScaler(X_yangben):
    sc = StandardScaler()
    sc.fit(X_yangben)
    return sc
def dataprocessing_second_StandardScaler_transform(_StandardScaler,X_yangben):
    X_yangben_std = _StandardScaler.transform(X_yangben)
    return X_yangben_std
def dataprocessing_third_split(X_yangben_std, Y_taget, _rate_split, _random_state):
    X_train, X_test, Y_train, Y_test = train_test_split(X_yangben_std, Y_taget, test_size=_rate_split, random_state=_random_state)
    return X_train, X_test, Y_train, Y_test
def dataprocessing_common(X_yangben, Y_taget, rate_split):
    smo = SMOTE(random_state=42)
    X_smo, y_smo = smo.fit_resample(X_yangben, Y_taget)
    sc = StandardScaler()
    sc.fit(X_smo)
    X_train_std = sc.transform(X_smo)
    X_yangben_std = sc.transform(X_yangben)
    poly_reg = PolynomialFeatures(degree=5)
    x_poly = poly_reg.fit_transform(X_train_std)
    X_yangben_poly = poly_reg.fit_transform(X_yangben_std)
    X_train, X_test, Y_train, Y_test = train_test_split(x_poly, y_smo, test_size=rate_split, random_state=0)
    return X_train, X_test, Y_train, Y_test, X_yangben_poly, Y_taget

def dataprocessing_LSTM(X_yangben, Y_taget, rate_split):
    smo = SMOTE(random_state=42)
    X_smo, y_smo = smo.fit_resample(X_yangben, Y_taget)
    sc = StandardScaler()
    sc.fit(X_smo)
    X_train_std = sc.transform(X_smo)
    X_yangben_std = sc.transform(X_yangben)
    X_train, X_test, Y_train, Y_test = train_test_split(X_train_std, y_smo, test_size=rate_split, random_state=0)
    return X_train, X_test, Y_train, Y_test, X_yangben_std, Y_taget


def statistics_major(masked_featureMap,nodata_value,num_multiply):
    masked_featureMap = masked_featureMap[np.logical_not(masked_featureMap == nodata_value)]
    masked_featureMap = masked_featureMap*num_multiply
    masked_featureMap = masked_featureMap.astype(int)
    bincounts = np.bincount(masked_featureMap)
    major = np.argmax(bincounts)
    major = float(major/num_multiply)
    return major

def statistics_mean(masked_featureMap,nodata_value):
    masked_featureMap = masked_featureMap[np.logical_not(masked_featureMap == nodata_value)]
    tem_mean = np.mean(masked_featureMap)
    return tem_mean

def statistics_gullymouth_mean(masked_featureMap, masked_flowAccu, scale_half_size, nodata_value):
    maximum = np.max(masked_flowAccu)
    index = np.where(masked_flowAccu == maximum)
    index = np.array(index).reshape([-1])
    m, n = masked_featureMap.shape
    sub_m_region, top_m_region = 0, 0
    sub_n_region, top_n_region = 0, 0
    if (index[0]-scale_half_size >= 0) and (index[0]+scale_half_size <= m-1):
        sub_m_region = index[0]-scale_half_size
        if sub_m_region<0:
            sub_m_region = 0
        top_m_region = index[0]+scale_half_size+1
    elif index[0]-scale_half_size < 0:
        sub_m_region = 0
        top_m_region = index[0]+scale_half_size*2+1
    else:
        sub_m_region = index[0]-scale_half_size*2
        if sub_m_region<0:
            sub_m_region = 0
        top_m_region = m+1
        
    if (index[1]-scale_half_size >= 0) and (index[1]+scale_half_size <= n-1):
        sub_n_region = index[1]-scale_half_size
        if sub_n_region<0:
            sub_n_region = 0
        top_n_region = index[1]+scale_half_size+1
    elif index[1]-scale_half_size < 0:
        sub_n_region = 0
        top_n_region = index[1]+scale_half_size*2+1
    else:
        sub_n_region = index[1]-scale_half_size*2
        if sub_n_region<0:
            sub_n_region = 0
        top_n_region = n+1

    gullymouth_region_featuremap = masked_featureMap[sub_m_region:top_m_region, sub_n_region:top_n_region]
    gullymouth_region_featuremap = gullymouth_region_featuremap[np.logical_not(gullymouth_region_featuremap == nodata_value)]
    if not(len(gullymouth_region_featuremap) > 0):
        tem_mean = 0.000001
    else:
        tem_mean = np.mean(gullymouth_region_featuremap)
    return tem_mean


if __name__ == '__main__':
    tem_matrix = [[1.0,4.0,1.0,1.0,1.0],
                  [1,2,8,5,1],
                  [1,3,7,4,1],
                  [1,3,4,6,1],
                  [1.0,3,1,1,1.0]]
    acc_matrix = [[1,4,1,1,1],
                  [1,2,8,5,1],
                  [100,3,1,4,1],
                  [1,3,4,6,1],
                  [1,3,1,1,1]]
    tem_matrix = np.array(tem_matrix)
    # tem_matrix = tem_matrix*10
    # tem_matrix = tem_matrix.astype(int)
    # print(tem_matrix[0:2,:])
    # mean = statistics_gullymouth_mean(tem_matrix, acc_matrix, 1, 1)
    # print(mean)
    nodata = 1

    major = statistics_major(tem_matrix,nodata,10000)
    print(major)