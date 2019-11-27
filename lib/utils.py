
from sklearn import preprocessing
import pandas as pd
import numpy as np



def pd_normalize(Data) :
    x = Data.values
    scaler = preprocessing.MinMaxScaler()
    x_n = scaler.fit_transform(x)
    data_n = pd.DataFrame(x_n, index=Data.index, columns=Data.columns)
    return data_n, scaler

def pd_one_hot_encoder(Data) :
    cols = Data.columns
    Ohe_df = pd.DataFrame(index=Data.index, columns=None)
    for col in Data.columns :
        one_hot = pd.get_dummies(Data[col])
        # Join the encoded df
        Ohe_df = pd.concat((Ohe_df, one_hot), axis='columns')

    return Ohe_df
