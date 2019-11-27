from lib.pd_load_data import *
from lib.analysis import *
from lib.utils import *
from lib.network_setup import *
from lib.evaluate_models import *
from lib.plot_results import *

import matplotlib.pyplot as plt
import statsmodels.api as sm

import PltOptions


if __name__ == '__main__' :
    # load the Data and output
    Data = load_data('data/test.csv', index_col='Id')
    Target = load_data('data/sample_submission.csv', index_col='Id')

    # Correlation analysis (NOT MANDATORY)
    features_nocorr = corr_all(Target, Data)
    Data.drop(features_nocorr, axis='columns', inplace=True)

    # Normalize continuous values
    continuous_df = Data.select_dtypes(include=(np.int64, int, float))
    continuous_n, scaler = pd_normalize(continuous_df)
    for col in continuous_n.columns :
        Data[col] = continuous_n[col]

    # Deal With Caterorical Data (One Hot encoding)
    categorical_df = Data.select_dtypes(include=(object))
    Ohe_df = pd_one_hot_encoder(categorical_df)
    Data.drop(columns=categorical_df.columns, inplace=True)
    Raw_input = pd.concat((Data, Ohe_df), axis='columns')


    # Create the Network that best suits the data
    #1# Linear Regression
    Target_Train, Target_Test = pd_split(Target, split=0.1)
    Raw_input_Train, Raw_input_test = pd_split(Raw_input, split=0.1)
    model = sm.OLS(Target_Train, Raw_input_Train)
    results = model.fit()
    OLS_plot_res(model, results, train=(Raw_input_Train, Target_Train), test=(Raw_input_test, Target_Test))

    #2# Classic NN Model
    model = load_NN(input_shape=len(Raw_input.columns), Output_shape=len(Target_Test.columns),
                    layers=4, n_units=100)
    history = model.fit(Raw_input_Train, Target_Train, epochs=200, batch_size=32, validation_split=0.1)
    NN_plot_res(model, history, train=(Raw_input_Train, Target_Train), test=None)
