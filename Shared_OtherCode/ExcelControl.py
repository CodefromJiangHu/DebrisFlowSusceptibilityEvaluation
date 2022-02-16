import pandas as pd
import numpy as np

def meanValueForList(numberList):
    sum_value = sum(numberList)
    avg = sum_value/len(numberList)
    return avg

def floatParseFromStr(StrList):
    temList = []
    for index in range(len(StrList)):
        temList.append(float(StrList[index]))
    return temList

def getValueByKeyValue(base_df, key_fieldName, key_value, return_filedName):
    result = ""
    df = base_df
    df_key = df[df[key_fieldName] == key_value]
    resultsList = df_key[return_filedName].tolist()
    if(len(resultsList)<1):
        result = "无"
    else:
        result = str(resultsList[0])
    return result

def getAvgRainfallIntenseByList(StrList):
    AvgRainfallList = []
    for index in range(len(StrList)):
        rainfallList = StrList[index].split(',')
        tem_rainfallList = floatParseFromStr(rainfallList)
        avg_value = meanValueForList(tem_rainfallList)
        AvgRainfallList.append( avg_value)
    return AvgRainfallList

def getAccumRainfallByList(StrList):
    AccumRainfallList = []
    for index in range(len(StrList)):
        rainfallList = StrList[index].split(',')
        tem_rainfallList = floatParseFromStr(rainfallList)
        sum_value = sum(tem_rainfallList)
        AccumRainfallList.append(sum_value)
    return AccumRainfallList


def convertDFtoArray(input_df, list_fieldsName):
    array = input_df.values
    tem_array = np.array(array)
    m, n = tem_array.shape
    # tem_2dArray = np.zeros(shape=[m, 1])
    final_array = np.zeros(shape=[m, len(list_fieldsName)])
    for i in range(len(list_fieldsName)):
        tem = np.array(input_df[list_fieldsName[i]].values).reshape(m)

        final_array[:, i] = tem
    return final_array

