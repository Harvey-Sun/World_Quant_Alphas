# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 11:05:02 2017

@author: LZJF
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
value = import_data('LZ_GPA_QUOTE_TVALUE')
volume = import_data('LZ_GPA_QUOTE_TVOLUME')
stop = import_data('LZ_GPA_SLCIND_STOP_FLAG')

# 日度指标
factor41 = ((high * low) ** 0.5 - (value / volume) * 10) #/ close
# factor41 = ((high * low) ** 0.5 - (value / volume) * 10) / (value / volume) * 10)
factor41 = pd.DataFrame(np.where(stop.isnull(), factor41, np.nan),
                        columns=high.columns, index=high.index)


factor_name = 'alpha#41_origin_1D'
factor41.index.name = factor_name
os.chdir('F:\Factors\World_Quant_Alphas\#41')
factor41.to_csv('%s.csv' % factor_name)

returns = (close * 0.999 - OPEN * 1.001) / (OPEN * 1.001)
group_return = [0]
for i in factor41.index[:-1]:
    group1 = factor41.ix[i].dropna().sort_values(ascending=True)[:100].index
    group_return.append(returns.shift(-1).ix[i][group1].mean())
group_return = pd.Series(group_return, index=returns.index[:-1])

(group_return['2013-01-01':] + 1).cumprod().plot()
(returns.mean(axis=1)['2013-01-01':] + 1).cumprod().plot()
(group_return + 1).cumprod().plot()
(returns.mean(axis=1) + 1).cumprod().plot()

(group_return['2013-01-01':].mean() + 1 - 0.002) ** (255 * 4)
