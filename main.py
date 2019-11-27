from lib.pd_load_data import *
from lib.analysis import *
from lib.utils import *
from Network_setup import *
import matplotlib.pyplot as plt


if __name__ == '__main__' :
    # load the Data and output
    Data = load_data('data/test.csv', index_col='Id')
    Pred = load_data('data/sample_submission.csv', index_col='Id')

    # Correlation analysis (NOT MANDATORY)
    features_nocorr = corr_all(Pred, Data)
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
