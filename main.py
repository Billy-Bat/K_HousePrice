from lib.pd_load_data import *
from lib.analysis import *
from lib.utils import *
import matplotlib.pyplot as plt


if __name__ == '__main__' :
    # load the Data and output
    Data = load_data('data/test.csv', index_col='Id')
    Pred = load_data('data/sample_submission.csv', index_col='Id')

    # Correlation analysis (NOT MANDATORY)
    features_nocorr = corr_all(Pred, Data)
    Data.drop(features_nocorr, axis='columns', inplace=True)

    # Normalize continuous values
    continuous_col = Data.select_dtypes(include=(np.int64, int, float))
    continuous_n, scaler = pd_normalize(continuous_col)
    for col in continuous_n.columns :
        Data[col] = continuous_n[col]
