
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

from math import floor, sqrt


import sys

def pd_split(Df, split=0.1) :
    index_l = len(Df.index)
    slice = int(floor(len(Df.index)*0.1))
    Train = Df.iloc[slice:,:]
    Test = Df.iloc[:slice,:]

    return Train, Test

def pd_normalize(Data) :
    x = Data.values
    x = np.nan_to_num(x, copy=True)
    scaler = preprocessing.MinMaxScaler()
    x_n = scaler.fit_transform(x)
    data_n = pd.DataFrame(x_n, index=Data.index, columns=Data.columns)
    return data_n, scaler

def pd_one_hot_encoder(Data) :
    cols = Data.columns
    Ohe_df = pd.DataFrame(index=Data.index, columns=None)
    for col in Data.columns :
        one_hot = pd.get_dummies(Data[col])
        Ohe_df = pd.concat((Ohe_df, one_hot), axis='columns')

    return Ohe_df

def measure_rmse(actual, predicted):
    eps = sys.float_info.epsilon
    return sqrt(mean_squared_error(actual, predicted+eps))
