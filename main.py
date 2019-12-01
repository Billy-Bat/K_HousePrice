from lib.pd_load_data import *
from lib.analysis import *
from lib.utils import *
from lib.network_setup import *
from lib.evaluate_models import *
from lib.plot_results import *

import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
from sklearn.model_selection import KFold


import PltOptions

numeric_dtypes = ('int16', 'int32', 'int64', 'float16', 'float32', 'float64')

if __name__ == '__main__' :
    # load the Data and output
    Data = load_data('data/train.csv', index_col='Id')
    Target = pd.DataFrame(Data[u'SalePrice'], index=Data.index)
    Target, lambdas_target = pd_boxcox(Target, rtrn_lambdas=True)
    Data = Data.drop(columns=[u'SalePrice'])
    missing_val_analysis(Target, Data, zeroes=False, _Save=True)



    # Apply Data Transformation continuous values
    continuous_df = Data.select_dtypes(include=numeric_dtypes)
    continuous_df = pd_fixskew(continuous_df,  exclude=('PoolArea', '3SsnPorch', 'LowQualFinSF', 'MiscVal', 'ScreenPorch')) # also standarzing the values
    skew_analysis(continuous_df, _Save=False)
    # continuous_rs, scaler = pd_robustscale(continuous_df)
    for col in continuous_rs.columns :
        Data[col] = continuous_rs[col]
    Target_rs, scaler = pd_robustscale(Target)

    # Deal With Caterorical Data (One Hot encoding)
    categorical_df = Data.select_dtypes(include=(object))
    Ohe_df = pd_one_hot_encoder(categorical_df)
    Data.drop(columns=categorical_df.columns, inplace=True)
    Raw_input = pd.concat((Data, Ohe_df), axis='columns')


    Target_Train, Target_Test = pd_split(Target, split=0.1)
    # # Create the Network that best suits the data
    # #1# Linear Regression
    # Raw_input_Train, Raw_input_test = pd_split(Raw_input, split=0.1)
    # model = sm.OLS(Target_Train, Raw_input_Train)
    # results = model.fit()
    # OLS_plot_res(model, results, train=(Raw_input_Train, Target_Train), test=(Raw_input_test, Target_Test))
    #
    # #2# Classic NN Model
    # model = load_NN(input_shape=len(Raw_input.columns), Output_shape=len(Target_Test.columns),
    #                 layers=4, n_units=100)
    # history = model.fit(Raw_input_Train, Target_Train, epochs=1000, batch_size=32, validation_split=0.1, verbose=0)
    # NN_plot_res(model, history, train=(Raw_input_Train, Target_Train), test=(Raw_input_test, Target_Test))

    #3# KNN Model
