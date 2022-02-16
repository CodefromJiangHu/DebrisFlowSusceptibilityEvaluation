import pandas as pd
import numpy as np
from collections import Counter

# removing outliers
def detect_outliers(df,n, features):
    outlier_indices = []
    for col in features:
        Q1 = np.percentile(df[col], 25)
        Q3 = np.percentile(df[col], 75)
        IQR = Q3 - Q1
        outlier_step = 1.5 * IQR
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step)].index
        print(outlier_list_col)
        outlier_indices.extend(outlier_list_col)
    outlier_indices = Counter(outlier_indices)
    print(outlier_indices)
    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n)
    return multiple_outliers
if __name__ == '__main__':
    list_fields_names = ["f_suscep", "f_dem", "f_ic",  "f_soiltrength", "f_vegload", "Gully_J_Sd", "流速Sd", "流深Sd"]
    df = pd.read_excel("Yangben_Details_Dadu_River_RO.xlsx")
    Outliers_to_drop = detect_outliers(df, 2, list_fields_names)
    print(Outliers_to_drop)
    df = df.drop(Outliers_to_drop, axis=0).reset_index(drop=True)
    df.to_excel("Yangben_Details_Dadu_River_RO_Output.xlsx")
