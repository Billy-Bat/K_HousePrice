import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from collections import OrderedDict
import scipy.stats as stats

import matplotlib.gridspec as gridspec

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
#     Correl = {}
#     for i, col in enumerate(Data.columns) :
#         first_index = Data[col].first_valid_index()
#         type_col = type(Data[col][first_index])
#
#         if  type_col == str :
#             Correl[col] = corr_categorical(Pred, Data[col], alpha=alpha)
#         elif ( type_col == float) | (type_col == int) | (type_col == np.int64) | (type_col == np.float64):
#             Correl[col] = corr_Pearson(Pred, Data[col], alpha=alpha)
#         else :
#             print('WARNING : column {} has unknown type'.format(col))
#
#     checked_features = []
#     for key in Correl :
#         if Correl[key] == False :
#             if disp :
#                 print('{} is not significant'.format(key))
#             checked_features.append(key)
#
    return checked_features

def pd_corr_Pearson(Pred, Data_col):
    res = 0

    index = Data_col.index[Data_col.apply(np.isnan)]
    Data_col = Data_col.drop(index)
    Pred = Pred.drop(index)
    if Data_col.isna().any() :
        print('WARNING : Dealing with NaN values')
    Pearson = Pred.corrwith(Data_col, method='pearson')
    r_value = Pearson.T[0]
    p_value = 0

    return r_value, p_value

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

def corr_heatmap(Pred, Data, methods=('pearson', 'spearman', 'kendall'), _Save=True):
    full = pd.concat((Pred, Data), axis=1, sort=True)

    fig = plt.figure(num=None, figsize=(3, 7), dpi=95, facecolor='w')
    gs = gridspec.GridSpec(len(methods), 2, width_ratios=[14, 1], wspace=0.2, hspace=0.5)

    result = []
    for i, method in enumerate(methods) :
        res = full.corr(method=method)

        axes = plt.subplot(gs[i,0])
        pc = plt.pcolor(res, cmap='afmhot')
        axes.set_title(method)
        axes.set_xticks(np.arange(res.shape[1]) + 0.5)
        axes.set_yticks(np.arange(res.shape[0]) + 0.5)
        axes.set_xticklabels(list(res.columns), rotation=90, fontdict={'weight': 'normal','size': 4})
        axes.set_yticklabels(list(res.index), fontdict={'weight': 'normal','size': 4})

        axes = plt.subplot(gs[i,1])
        plt.colorbar(pc, cax=axes)

    if _Save :
        fig = plt.gcf()
        fig.savefig('vizu/CorrelHeatMap.png')

    plt.show()

    return 0

def missing_val_analysis(Pred, Data, _Save=False) :
    full = pd.concat((Pred, Data), axis=1, sort=True)
    col_with_na = {}

    for col in full.columns :
        if full[col].isna().any() :
            n = full[col].isna().sum(skipna=False)
            pct_col = float(n)/len(full[col])
            col_with_na[col] = pct_col

    ordered = OrderedDict(sorted(col_with_na.items(), key=lambda item:np.min(item[1])))
    cols = list(ordered.keys())
    values = [x*100 for x in ordered.values()]


    bar_plot = plt.bar(x=cols, height=values, align='center')
    ax = plt.gca()
    ax.set_title('Missing values (NaN)', fontdict={'weight': 'normal','size': 12})
    ax.set_xlabel('Column Labels', fontdict={'weight': 'normal','size': 12})
    ax.set_ylabel('missing values (in %)', fontdict={'weight': 'normal','size': 12})
    plt.xticks(rotation=70)

    if _Save :
        fig = plt.gcf()
        fig.savefig('vizu/MissingValues.png')


    plt.show()

    return 1

def outlier_analysis(col, Pred, Data, tresh=0.99, _Save=False):
    data = pd.concat((Pred, Data[col]), axis=1, sort=True)
    Pred_quant = Pred.quantile(tresh).values[0]
    Data_quant = Data[col].quantile(tresh)

    s1 = sns.jointplot(x=col, y=Pred.columns[0], data=data, kind = "reg")
    s1.ax_joint.plot([0, Data[col].max()*1.1], [Pred_quant, Pred_quant], linewidth=2)
    s1.ax_joint.plot([Data_quant, Data_quant], [0, Pred.max()*1.1], linewidth=2)
    s1.set_axis_labels(xlabel=col, ylabel=Pred.columns[0])
    s1.annotate(stats.pearsonr)

    if _Save :
        fig = plt.gcf()
        fig.savefig('vizu/OutlierAnalysis_{}.png'.format(col))

    plt.show()

    return 0
