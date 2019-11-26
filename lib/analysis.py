import pandas as pd
import numpy as np



def corr_all(Pred, Data):
    temp = []
    for i, col in enumerate(Data.columns) :
        first_index = Data[col].first_valid_index()
        type_col = type(Data[col][first_index])

        if  type_col == str :
            corr_LDA(Pred, Data[col])
        elif ( type_col == float) | (type_col == int) | (type_col == np.int64) | (type_col == np.float64):
            corr_Pearson(Pred, Data[col])
        else :
            print('WARNING : column {} has unknown type'.format(col))
    return 0

def corr_Pearson(Pred, Data_col):
    return 0

def corr_LDA(Pred, Data_col) :
    return 0
