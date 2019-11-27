
from sklearn import preprocessing
import pandas as pd



def pd_normalize(Data) :
    x = Data.values
    scaler = preprocessing.MinMaxScaler()
    x_n = scaler.fit_transform(x)
    data_n = pd.DataFrame(x_n, index=Data.index, columns=Data.columns)
    return data_n, scaler
