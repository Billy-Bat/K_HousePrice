import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


def normally_distributed(Data, alpha=0.05, _plot=True, _Save=True) :
    stat, p = stats.normaltest(Data)
    res = None
    if p > alpha:
       print('Sample follows Gaussiann, p = {}, alpha = {}'.format(p, alpha))
       res = 1
    else:
       print('Sample does not follow Gaussian, p = {}, alpha = {}'.format(p, alpha))
       res = 0

    if _plot :
        ax = Data.plot.density()
        ax.set_title('p = {}, alpha = {}'.format(p, alpha))

    if _Save :
        fig = plt.gcf()
        fig.savefig('vizu/distributionPlot.png')

    plt.show()
    return res


def corr_all(Pred, Data, alpha=0.05, disp=False):
    Correl = {}
    for i, col in enumerate(Data.columns) :
        first_index = Data[col].first_valid_index()
        type_col = type(Data[col][first_index])

        if  type_col == str :
            Correl[col] = corr_categorical(Pred, Data[col], alpha=alpha)
        elif ( type_col == float) | (type_col == int) | (type_col == np.int64) | (type_col == np.float64):
            Correl[col] = corr_Pearson(Pred, Data[col], alpha=alpha)
        else :
            print('WARNING : column {} has unknown type'.format(col))

    checked_features = []
    for key in Correl :
        if Correl[key] == False :
            if disp :
                print('{} is not significant'.format(key))
            checked_features.append(key)

    return checked_features

def corr_Pearson(Pred, Data_col, alpha):
    res = 0

    index = Data_col.index[Data_col.apply(np.isnan)]
    Data_col = Data_col.drop(index)
    Pred = Pred.drop(index)
    if Data_col.isna().any() :
        print('WARNING : Dealing with NaN values')
    Pearson = Pred.corrwith(Data_col, method='pearson')
    r_value = Pearson.T[0]

    if abs(r_value) > alpha :
        res = 1
    else :
        res = 0

    return res

def corr_categorical(Pred, Data_col, alpha, _Gaussian=False) :
    res = 0

    Data_col.fillna(value='None', inplace=True)
    Tags = Data_col.unique()
    dic_tags = {}

    for tag in Tags :
        tag_index = Data_col.index[Data_col == tag].tolist()
        tag_prices = Pred.ix[tag_index].values
        dic_tags[tag] = tag_prices

    if _Gaussian : # Normally ditributed <--- ANOVA
        print('WIP : Anova test not implemented')
        # OW_ANOVA = stats.f_oneway(Data_col, Pred)
        return 0
    else : # Otherwise <--- Kruskal Wallis test
        tuple_arg = ([x for x in list(dic_tags.values())])
        kruskal_res = stats.kruskal(*tuple_arg)
        p_value, H_value = (kruskal_res.pvalue, kruskal_res.statistic)
        if p_value < alpha :
            res = 1
        else :
            res = 0

    return res
