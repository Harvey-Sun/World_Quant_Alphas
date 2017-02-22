# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 17:38:39 2017

@author: LZJF
"""

import pandas as pd
import numpy as np
import matplotlib as mpl
# import statsmodels.api as sm
import os
mpl.style.use('ggplot')

os.chdir('C:\cStrategy\Factor')


def import_data(name):
    data = pd.read_csv('%s.csv' % name, index_col=0)
    data.index = pd.to_datetime([str(i) for i in data.index])
    return data


def pearson(x, y, skipna=False):
    upside = ((x - x.mean()) * (y - y.mean())).sum(skipna=skipna)
    downside = np.sqrt((np.square(x - x.mean()).sum()) *
                       (np.square(y - y.mean()).sum(skipna=skipna)))
    return upside / downside


def norm_rank(data):
    ranked = data.rank(axis=1)
    norm_ranked = (ranked.T / ranked.max(axis=1)).T
    return norm_ranked


def save_factor(factor, factor_id):
    factor_name = 'alpha#%s_origin_1D' % factor_id
    factor.index.name = factor_name
    factor.to_csv('F:\Factors\World_Quant_Alphas\#%s\%s.csv' %
                  (factor_id, factor_name))


def factor_cal(data, factor_id):
    high = data['high']
    low = data['low']
    close = data['close']
    OPEN = data['OPEN']
    value = data['value']
    volume = data['volume']
    stop = data['stop']
    if factor_id == '002':
        temp1 = np.log(volume).diff(2)
        temp1_rank = norm_rank(temp1)
        temp2 = (close - OPEN) / OPEN
        temp2_rank = norm_rank(temp2)
        factor = pd.DataFrame()
        for t in temp1_rank.index:
            day_factor = -1 * pearson(temp1_rank[:t][-6:], temp2_rank[:t][-6:])
            factor[t] = day_factor
        factor = factor.T
    elif factor_id == '041':
        factor = ((high * low) ** 0.5 - (value / volume) * 10)
        factor = pd.DataFrame(np.where(stop.isnull(), factor, np.nan),
                              columns=high.columns, index=high.index)
    elif factor_id == '055':
        numerator = close - low.rolling(window=12).min()
        denominator = high.rolling(window=12).max() - low.rolling(window=12).min()
        fraction = numerator / denominator
        fraction_rank = fraction.rank(axis=1)
        volume_rank = volume.rank(axis=1)
        factor = pd.DataFrame()
        for t in fraction_rank.index:
            day_factor = -1 * pearson(fraction_rank[:t][-6:],
                                      volume_rank[:t][-6:])
            day_factor.name = t
            factor[t] = day_factor
        factor = factor.T
    elif factor_id == '101':
        factor = (close - OPEN) / ((high - low) + 0.001)
        factor = pd.DataFrame(np.where(stop.isnull(), factor, np.nan),
                              columns=factor.columns, index=factor.index)
    save_factor(factor, factor_id)
    return factor


def factor_summary(factor, period=1):
    percentiles = np.arange(0.1, 1, 0.1)
    summary = pd.DataFrame()
    for i in range(0, len(factor.index), period):
        t = factor.index[i]
        summary[t] = factor.ix[t].dropna().describe(percentiles)
    return summary.T


def ic(factor, return_mode, period, data, by_day=False):
    close = data['close']
    OPEN = data['OPEN']
    if return_mode == 'close-close':
        returns = close.pct_change(period).shift(-period).stack()
    elif return_mode == 'close-open':
        returns = (close.shift(-(period-1)) / OPEN - 1).shift(-period).stack()
    elif return_mode == 'open-open':
        returns = OPEN.pct_change(period).shift(-(period + 1)).stack()
    else:
        print 'mode must be \'close-close\', \'close-open\', \'open-open\''
    returns.name = 'returns'
    factor = factor.stack()
    factor.name = 'factor'
    temp = pd.concat([returns, factor], axis=1).dropna(axis=0, how='any')
    if by_day:
        ic_list = []
        days = temp.index.levels[0]
        for t in days:
            ic_list.append(temp.ix[t].corr().iloc[0][1])
        ic_series = pd.Series(ic_list, index=days)
        return ic_series
    else:
        return temp.corr().iloc[0][1]


def group_analysis(factor, data, return_mode, period=1, bins=10):
    close = data['close']
    OPEN = data['OPEN']
    if return_mode == 'close-close':
            returns = close.pct_change(period).shift(-period)
    elif return_mode == 'close-open':
        returns = (close.shift(-(period-1)) / OPEN - 1).shift(-period)
    elif return_mode == 'open-open':
        returns = OPEN.pct_change(period).shift(-(period + 1))
    else:
        print 'mode must be \'close-close\', \'close-open\', \'open-open\''
    days = len(factor.index)
    group_mean = pd.DataFrame()
    group_return = pd.DataFrame()
    for i in range(0, days, period):
        t = factor.index[i]
        temp = factor.ix[t].dropna()
        temp.sort_values(ascending=True, inplace=True)
        temp[:] = range(len(temp))
        # 根据实际数值大小分组，可能某一个数值的股票数量太多，两个bin的边界相同
        # 这里把因子排序，并用一个从1开始的序列替代原数值，在分组较多时，可能不准确
        groups = pd.qcut(temp, bins, labels=False)
        group_mean[t] = temp.groupby(groups).mean()
        group_return[t] = returns.ix[t].dropna().groupby(groups).mean()
    a_return = returns.mean(axis=1)
    # group_mean.index = group_mean.index + 1
    # group_return.index = group_return.index + 1
    group_mean.index.name = 'Group'
    group_return.index.name = 'Group'
    group_return = group_return.T
    group_return['All_mean'] = a_return
    return {'group_mean': group_mean.T, 'group_return': group_return}


def _analysis(factor, data, return_mode, stock_number, period=1):
    close = data['close']
    OPEN = data['OPEN']
    if return_mode == 'close-close':
            returns = close.pct_change(period).shift(-period)
    elif return_mode == 'close-open':
        returns = (close.shift(-(period-1)) / OPEN - 1).shift(-period)
    elif return_mode == 'open-open':
        returns = OPEN.pct_change(period).shift(-(period + 1))
    else:
        print 'mode must be \'close-close\', \'close-open\', \'open-open\''
    days = len(factor.index)
    _mean = []
    _return = []
    _t = []
    for i in range(0, days, period):
        t = factor.index[i]
        _t.append(t)
        temp = factor.ix[t].dropna()
        temp.sort_values(ascending=True, inplace=True)
        stock_to_hold = temp[:stock_number].index
        _return.append(returns.ix[t][stock_to_hold].mean())
        _mean.append(temp[stock_to_hold].mean())
    _mean = pd.Series(_mean, index=_t, name='factor_mean')
    _return = pd.Series(_return, index=_t, name='mean_return')
    return {'factor_mean': _mean, 'mean_return': _return}


high = import_data('LZ_GPA_QUOTE_THIGH')
low = import_data('LZ_GPA_QUOTE_TLOW')
close = import_data('LZ_GPA_QUOTE_TCLOSE')
OPEN = import_data('LZ_GPA_QUOTE_TOPEN')
value = import_data('LZ_GPA_QUOTE_TVALUE')
volume = import_data('LZ_GPA_QUOTE_TVOLUME')
stop = import_data('LZ_GPA_SLCIND_STOP_FLAG')

data = {'high': high,
        'low': low,
        'close': close,
        'OPEN': OPEN,
        'value': value,
        'volume': volume,
        'stop': stop}

factor_041 = factor_cal(data, '041')

summary = factor_summary(factor_041['2013-01-01':])
summary.ix[:, ['mean', 'std', '10%', '90%']].plot(figsize=(10, 10))

ic_series = ic(factor_041, 'close-close', 1, data, by_day=True)
title = '60 days moving average of IC (absolute value)'
np.abs(ic_series).rolling(window=60).mean().plot(figsize=(14, 7), title=title)

temp = factor_041['2013-01-01':]
t = temp.index[0]
groups = pd.qcut(temp.ix[t].dropna(), 10, labels=False)
temp.ix[t].dropna().groupby(groups).mean()
close.pct_change().shift(-1).ix[t].dropna().groupby(groups).mean()

stt = factor_041['2013-01-01':].replace(0, np.nan).dropna(axis=0, how='all')

a = group_analysis(stt, data, 'close-close', period=1, bins=40)
a['group_return'].mean().sort_values()
a['group_mean'].min()
(a['group_return'] + 1).cumprod().plot(figsize=(14, 7))


a = _analysis(stt, data, 'close-close', stock_number=50)
a['mean_return'].mean()
a['factor_mean'].min()
(a['mean_return'] + 1).cumprod().plot(figsize=(14, 7))
