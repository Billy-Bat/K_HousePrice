from lib.pd_load_data import *
from lib.analysis import *
import matplotlib.pyplot as plt


if __name__ == '__main__' :
    Data = load_data('data/test.csv', index_col='Id')
    # types =  {u'MSZoning': str, 'MSZoning':str, 'Street': str ,'Alley': str , 'LotShape':int , 'LandContour': int, 'Utilities': , 'LotConfig': , 'LandSlope': , 'Neighborhood': , 'Condition1': ,
    #           'Condition2': , 'BldgType': , 'HouseStyle': , 'RoofStyle': , 'RoofMatl': , 'Exterior1st': , 'Exterior1st': , 'MasVnrType': , 'ExterQual': , 'ExterQual' :, 'ExterCond': ,
    #           'Foundation': , 'BsmtQual':, 'BsmtCond': , 'BsmtExposure': , 'BsmtFinType1':, 'BsmtFinType2': , 'Heating': , 'HeatingQC': , 'CentralAir': , 'Electrical' :, 'KitchenQual': ,
    #           'Functional': , 'FireplaceQu':, 'GarageType': , 'GarageFinish': , 'GarageQual': , 'PavedDrive':, 'PoolQC': , 'Fence':, 'MiscFeature': , 'SaleType': , 'SaleCondition': }
    # Data = Data.astype(dtype = types)

    # get the good types of data



    Pred = load_data('data/sample_submission.csv', index_col='Id')
    # normally_distributed(Pred)
    corr_all(Pred, Data)
