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
factor = (close - OPEN) / ((high - low) + 0.001)
factor = pd.DataFrame(np.where(stop.isnull(), factor, np.nan),
                      columns=factor.columns, index=factor.index)

factor_name = 'alpha#101_origin_1D'
factor.index.name = factor_name
os.chdir('F:\Factors\World_Quant_Alphas\#101')
factor.to_csv('%s.csv' % factor_name)


returns = (close * 0.998 - OPEN * 1.001) / (OPEN * 1.001)
group_return = [0]
for i in factor.index[:-1]:
    group1 = factor.ix[i].dropna().sort_values(ascending=False)[:100].index
    group_return.append(returns.shift(-1).ix[i][group1].mean())
group_return = pd.Series(group_return, index=returns.index)

(group_return['2013-01-01':] + 1).cumprod().plot()
(returns.mean(axis=1)['2013-01-01':] + 1).cumprod().plot()
(group_return + 1).cumprod().plot()
(returns.mean(axis=1) + 1).cumprod().plot()

(group_return['2013-01-01':].mean() + 1 - 0.002) ** (255 * 4)
