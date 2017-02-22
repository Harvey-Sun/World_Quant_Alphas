# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 11:05:02 2017

@author: LZJF
"""

import pandas as pd
import numpy as np
import os

os.chdir('C:\cStrategy\Factor')


def import_data(name):
    data = pd.read_csv('%s.csv' % name, index_col=0)
    data.index = pd.to_datetime([str(i) for i in data.index])
    return data

high = import_data('LZ_GPA_QUOTE_THIGH')
low = import_data('LZ_GPA_QUOTE_TLOW')
close = import_data('LZ_GPA_QUOTE_TCLOSE')
OPEN = import_data('LZ_GPA_QUOTE_TOPEN')
volume = import_data('LZ_GPA_QUOTE_TVOLUME')
stop = import_data('LZ_GPA_SLCIND_STOP_FLAG')

# 日度指标
numerator = close - low.rolling(window=12).min()
denominator = high.rolling(window=12).max() - low.rolling(window=12).min()
fraction = numerator / denominator
fraction_rank = fraction.rank(axis=1)
volume_rank = volume.rank(axis=1)


def pearson(x, y, skipna=False):
    N = len(x)
    sumofx = x.sum(skipna=skipna)
    sumofy = y.sum(skipna=skipna)
    sumofxy = (x * y).sum(skipna=skipna)
    sumofxx = (x * x).sum(skipna=skipna)
    sumofyy = (y * y).sum(skipna=skipna)
    upside = sumofxy - sumofx * sumofy / N
    downside = np.sqrt((sumofxx - (sumofx ** 2) / N) *
                       (sumofyy - (sumofy ** 2) / N))
    return upside / downside

factor = pd.DataFrame()
for t in fraction_rank.index:
    day_factor = -1 * pearson(fraction_rank[:t][-6:], volume_rank[:t][-6:])
    day_factor.name = t
    factor[t] = day_factor
factor = factor.T

factor_name = 'alpha#55_origin_1D'
factor.index.name = factor_name
os.chdir('F:\Factors\World_Quant_Alphas\#55')
factor.to_csv('%s.csv' % factor_name)


returns = (close * 0.998 - OPEN * 1.001) / (OPEN * 1.001)
group_return = [0]
for i in factor.index[18:-1]:
    group1 = factor.ix[i].dropna().sort_values(ascending=False)[:100].index
    group_return.append(returns.shift(-1).ix[i][group1].mean())
group_return = pd.Series(group_return, index=returns.index[18:])

(group_return['2013-01-01':] + 1).cumprod().plot()
(returns.mean(axis=1)['2013-01-01':] + 1).cumprod().plot()
(group_return + 1).cumprod().plot()
(returns.mean(axis=1) + 1).cumprod().plot()

(group_return['2013-01-01':].mean() + 1 - 0.002) ** (255 * 4)
