import warnings
warnings.filterwarnings("ignore")

from lib.pd_load_data import *
from lib.analysis import *
from lib.utils import *
from lib.network_setup import *
from lib.evaluate_models import *
from lib.plot_results import *
import PltOptions

import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np


## # TEMP:
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from keras.wrappers.scikit_learn import KerasClassifier
import lightgbm as lgb

numeric_dtypes = ('int16', 'int32', 'int64', 'float16', 'float32', 'float64')
col_num2binary = ('Fireplaces', 'WoodDeckSF', 'EnclosedPorch', 'PoolArea',
                  '3SsnPorch', 'LowQualFinSF', 'MiscVal', 'ScreenPorch', 'BsmtFullBath',
                  'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr')

if __name__ == '__main__' :
    # load the Data and output
    Data = load_data('data/train.csv', index_col='Id')
    Target = pd.DataFrame(Data[u'SalePrice'], index=Data.index)
    Target, lambdas_target = pd_boxcox(Target, rtrn_lambdas=True)
    Data = Data.drop(columns=[u'SalePrice'])

    # Apply Data Transformation continuous values
    continuous_df = Data.select_dtypes(include=numeric_dtypes)
    continuous_df = df_toBinary(continuous_df, include=col_num2binary)
    continuous_df = pd_fixskew(continuous_df,  exclude=col_num2binary) # also standarzing the values
    for col in continuous_df.columns :
        Data[col] = continuous_df[col]
    Target_rs, transformer = pd_fixskew(Target, return_lambda=True)

    # Deal With Caterorical Data (One Hot encoding)
    categorical_df = Data.select_dtypes(include=(object))
    Ohe_df = pd_one_hot_encoder(categorical_df)
    Data.drop(columns=categorical_df.columns, inplace=True)
    Raw_input = pd.concat((Data, Ohe_df), axis='columns')
    # Raw_input.drop(axis=1, columns=['LotFrontage', 'MasVnrArea', 'GarageYrBlt'], inplace=True)


    ###############################Evaluate Models#############################
    # Lasso model
    lasso = make_pipeline(RobustScaler(), Lasso(alpha=0.0005, random_state=1))
    score = rmsle_cv(lasso, Raw_input, Target_rs, n_folds=5)
    print('lasso score : {} ({})'.format(score.mean(), score.std()))

    # ElasticNet
    ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
    score = rmsle_cv(ENet, Raw_input, Target_rs, n_folds=5)
    print('ENet score : {} ({})'.format(score.mean(), score.std()))

    # SVR
    svr = make_pipeline(RobustScaler(), SVR(C= 20, epsilon= 0.008, gamma=0.0003,))
    score = rmsle_cv(svr, Raw_input, Target_rs, n_folds=5)
    print('SVR score : {} ({})'.format(score.mean(), score.std()))

    # KernelRidge
    KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
    score = rmsle_cv(KRR, Raw_input, Target_rs, n_folds=5)
    print('KernelRidge score : {} ({})'.format(score.mean(), score.std()))

    # GradBoost
    GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10,
                                   loss='huber', random_state =5)
    print('GradBoost score : {} ({})'.format(score.mean(), score.std()))

    # LGBMRegressor
    lgb_model = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
    score = rmsle_cv(lgb_model, Raw_input, Target_rs, n_folds=5)
    print('LGBMRegressor score : {} ({})'.format(score.mean(), score.std()))


    # NN model
    # neural_network = KerasClassifier(build_fn=load_NN,
    #                              epochs=200,
    #                              batch_size=100,
    #                              verbose=0)
    #
    # # NN_model = load_NN(input_shape=Raw_input.shape[0], Output_shape=1, layers=5, n_units=100)
    # score = rmsle_cv(neural_network, Raw_input, Target_rs, n_folds=5)
    # print('NN score : {} ({})'.format(score.mean(), score.std()))

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
