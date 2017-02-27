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


def save_factor(factor, factor_id):
    factor_name = 'alpha#%s_origin_1D' % factor_id
    factor.index.name = factor_name
    to_file = 'F:\Strategies\World_Quant_Alphas/#%s' % factor_id
    if not os.path.exists(to_file):
        os.mkdir(to_file)
    else:
        pass
    factor.to_csv(to_file + '/%s.csv' % factor_name)
    print '保存完成'


def rolling_stop(stop, window):
    temp = pd.DataFrame(index=stop.columns)
    for i in stop.index:
        one = stop[:i][-window:].notnull().any(axis=0)
        temp[i] = one
    return temp.T


def rolling_rank(data, window):
    temp = pd.DataFrame(index=data.columns)
    for i in data.index:
        one = data[:i][-window:].rank(axis=0, pct=True).iloc[-1]
        temp[i] = one
    return temp.T


def factor_cal(data, factor_id, no_zdt=False):
    high = data['high']
    low = data['low']
    close = data['close']
    OPEN = data['OPEN']
    value = data['value']
    volume = data['volume']
    stop = data['stop']
    price_adj_f = data['price_adj_f']
    adj_OPEN = OPEN * price_adj_f
    adj_close = close * price_adj_f
    adj_high = high * price_adj_f
    adj_low = low * price_adj_f
    adj_volume = volume / price_adj_f
    returns = adj_close.pct_change()
    if factor_id == '001':
        returns = adj_close.pct_change()
        returns[returns < 0] = returns.rolling(window=20).std()
        ts_argmax = np.square(returns).rolling(window=5).apply(np.argmax) + 1
        factor = ts_argmax.rank(axis=1, pct=True) - 0.5
        factor[rolling_stop(stop, 20)] = np.nan
    elif factor_id == '002':
        temp1 = np.log(adj_volume).diff(2)
        temp1_rank = temp1.rank(axis=1, pct=True)
        temp2 = (close - OPEN) / OPEN
        temp2_rank = temp2.rank(axis=1, pct=True)
        factor = pd.DataFrame()
        for t in temp1_rank.index:
            day_factor = -1 * pearson(temp1_rank[:t][-6:], temp2_rank[:t][-6:])
            factor[t] = day_factor
        factor = factor.T
    elif factor_id == '003':
        open_rank = OPEN.rank(axis=1, pct=True)
        volume_rank = volume.replace(0, np.nan).rank(axis=1, pct=True)
        factor = pd.DataFrame()
        for t in open_rank.index:
            day_factor = -1 * pearson(open_rank[:t][-10:], volume_rank[:t][-10:])
            factor[t] = day_factor
        factor = factor.T
        factor[stop.notnull()] = np.nan
    elif factor_id == '004':
        low_rank = adj_low.rank(axis=1, pct=True)
        factor = -1 * rolling_rank(low_rank, 9)
        factor[rolling_stop(stop, 9)] = np.nan
    elif factor_id == '005':
        vwap = value / adj_volume
        be_half = (adj_OPEN - vwap.rolling(window=10).mean()).rank(axis=1, pct=True)
        af_half = (adj_close - vwap).rank(axis=1, pct=True)
        factor = -1 * be_half * af_half
    elif factor_id == '006':
        factor = pd.DataFrame()
        adj_volume = adj_volume.replace(0, np.nan)
        for t in OPEN.index:
            day_factor = -1 * pearson(adj_OPEN[:t][-10:], adj_volume[:t][-10:])
            factor[t] = day_factor
        factor = factor.T
    elif factor_id == '007':
        factor = -1 * rolling_rank(np.abs(adj_close.diff(7)), 60) * np.sign(adj_close.diff(7))
        factor[adj_volume.rolling(window=20).mean() > volume] = -1
    elif factor_id == '008':
        temp = OPEN.rolling(window=5).sum() * returns.rolling(window=5).sum()
        factor = -1 * (temp - temp.diff(10)).rank(axis=1, pct=True)
        # temp - temp.diff(10) 不就是10日前的temp吗？ SB啊！
        factor[stop.notnull()] = np.nan
    elif factor_id == '009':
        cond_1 = adj_close.diff().rolling(window=5).min() > 0
        cond_2 = adj_close.diff().rolling(window=5).max() < 0
        factor = -1 * adj_close.diff()
        factor[cond_1 | cond_2] = adj_close.diff()
    elif factor_id == '010':
        cond_1 = adj_close.diff().rolling(window=4).min() > 0
        cond_2 = adj_close.diff().rolling(window=4).max() < 0
        factor = -1 * adj_close.diff()
        factor[cond_1 | cond_2] = adj_close.diff()
    elif factor_id == '011':
        vwap = value / adj_volume
        temp1 = (vwap - adj_close).rolling(window=3).max().rank(aixs=1, pct=True)
        temp2 = (vwap - adj_close).rolling(window=3).min().rank(axis=1, pct=True)
        temp3 = adj_volume.diff(3).rank(axis=1, pct=True)
        factor = (temp1 + temp2) * temp3
        factor[rolling_stop(stop, 3)] = np.nan
    elif factor_id == '012':
        factor = -1 * np.sign(adj_volume.diff()) * adj_close.diff()
        factor[stop.notnull()] = np.nan
    elif factor_id == '013':
        close_rank = adj_close.rank(axis=1, pct=True)
        volume_rank = adj_volume.rank(axis=1, pct=True)
        factor = -1 * (close_rank.rolling(window=5).cov(volume_rank)).rank(axis=1, pct=True)
        factor[rolling_stop(stop, 5)] = np.nan
    elif factor_id == '014':
        temp1 = adj_OPEN.rolling(window=10).corr(adj_volume)
        temp2 = returns.diff(3).rank(axis=1, pct=True)
        factor = -1 * temp2 * temp1
        factor[rolling_stop(stop, 10)] = np.nan
    elif factor_id == '015':
        high_rank = adj_high.rank(axis=1, pct=True)
        volume_rank = adj_volume.rank(axis=1, pct=True)
        temp1 = high_rank.rolling(window=3).corr(volume_rank)
        factor = -1 * temp1.rank(axis=1, pct=True).rolling(window=3).sum()
        factor[rolling_stop(stop, 3)] = np.nan
    elif factor_id == '041':
        factor = ((high * low) ** 0.5 - (value / volume) * 10)
        factor[stop.notnull()] = np.nan
    elif factor_id == '055':
        numerator = adj_close - adj_low.rolling(window=12).min()
        denominator = adj_high.rolling(window=12).max() - adj_low.rolling(window=12).min()
        fraction = numerator / denominator
        fraction[rolling_stop(stop, 12)] = np.nan
        fraction_rank = fraction.rank(axis=1, pct=True)
        volume_rank = volume.replace(0, np.nan).rank(axis=1, pct=True)
        factor = pd.DataFrame()
        for t in fraction_rank.index:
            day_factor = -1 * pearson(fraction_rank[:t][-6:],
                                      volume_rank[:t][-6:])
            day_factor.name = t
            factor[t] = day_factor
        factor = factor.T
    elif factor_id == '101':
        factor = (close - OPEN) / ((high - low) + 0.001)
        factor[stop.notnull()] = np.nan
    elif factor_id == 'vol':
        returns = adj_close.pct_change()
        factor = pd.DataFrame()
        for t in returns.index:
            not_stop = stop[:t][-21:].isnull().all(axis=0)
            vol = adj_close[:t][-21:].pct_change()[1:].std(skipna=False)
            vol = pd.Series(np.where(not_stop, vol, np.nan), index=not_stop.index)
            vol.index.name = t
            factor[t] = vol
        factor = factor.T
    else:
        print 'factor id is wrong'
    # 将涨跌停股票的因子设为na
    if no_zdt:
        increase_stop = np.round(close * 1.1, 2).shift(1)
        decrease_stop = np.round(close * 0.9, 2).shift(1)
        bool_temp = np.logical_or(close == increase_stop, decrease_stop == close)
        factor = np.where(bool_temp, np.nan, factor)
        factor = pd.DataFrame(factor, columns=close.columns, index=close.index)
    print '计算完成，保存因子文件'
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
    price_adj_f = data['price_adj_f']
    adj_close = close * price_adj_f
    adj_OPEN = OPEN * price_adj_f
    if return_mode == 'close-close':
        returns = adj_close.pct_change(period).shift(-period).stack()
    elif return_mode == 'close-open':
        returns = (adj_close.shift(-(period-1)) / adj_OPEN - 1).shift(-period).stack()
    elif return_mode == 'open-open':
        returns = adj_OPEN.pct_change(period).shift(-(period + 1)).stack()
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


def group_analysis(factor, data, return_mode, period=1, bins=10, cut_mode='quantile'):
    close = data['close']
    OPEN = data['OPEN']
    price_adj_f = data['price_adj_f']
    adj_close = close * price_adj_f
    adj_OPEN = OPEN * price_adj_f
    if return_mode == 'close-close':
        returns = adj_close.pct_change()
    # elif return_mode == 'close-open':
    #    returns = (adj_close.shift / adj_OPEN - 1)
    elif return_mode == 'open-open':
        returns = adj_OPEN.pct_change()
    else:
        print 'mode must be \'close-close\', \'close-open\', \'open-open\''
    days = len(factor.index)
    group_mean = pd.DataFrame()
    group_return = pd.DataFrame()
    for i in range(0, days, period):
        t = factor.index[i]
        temp = factor.ix[t].dropna()
        if cut_mode == 'quantile':
            temp.sort_values(ascending=True, inplace=True)
            temp[:] = range(len(temp))
            # 根据实际数值大小分组，可能某一个数值的股票数量太多，两个bin的边界相同
            # 这里把因子排序，并用一个从1开始的序列替代原数值，在分组较多时，可能不准确
            groups = pd.qcut(temp, bins, labels=False)
        elif cut_mode == 'interval':
            groups = pd.cut(temp, bins, labels=False)
        group_mean[t] = factor.ix[t].dropna().groupby(groups).mean()
        if return_mode == 'close-close':
            period_returns = returns.ix[t:].ix[1: period+1].groupby(groups, axis=1).mean()
        elif return_mode == 'open-open':
            period_returns = returns.ix[t:].ix[2: period+2].groupby(groups, axis=1).mean()
        group_return = pd.concat([group_return, period_returns])
    a_return = returns.mean(axis=1)
    a_mean = factor.mean(axis=1)
    # group_mean.index = group_mean.index + 1
    # group_return.index = group_return.index + 1
    group_mean.index.name = 'Group'
    group_mean = group_mean.T
    group_mean['Factor Average'] = a_mean
    group_return.columns.name = 'Group'
    group_return = group_return
    group_return['Simple Average'] = a_return
    return {'group_mean': group_mean, 'group_return': group_return}


def _analysis(factor, data, return_mode, stock_number, ascending=True, period=1):
    close = data['close']
    OPEN = data['OPEN']
    price_adj_f = data['price_adj_f']
    adj_close = close * price_adj_f
    adj_OPEN = OPEN * price_adj_f
    if return_mode == 'close-close':
        returns = adj_close.pct_change(period).shift(-period)
    elif return_mode == 'close-open':
        returns = (adj_close.shift(-(period-1)) / adj_OPEN - 1).shift(-period)
    elif return_mode == 'open-open':
        returns = adj_OPEN.pct_change(period).shift(-(period + 1))
    else:
        print 'mode must be \'close-close\', \'close-open\', \'open-open\''
    days = len(factor.index)
    _mean = []
    _return = []
    _t = []
    stocks = pd.DataFrame()
    factors = pd.DataFrame()
    for i in range(0, days, period):
        t = factor.index[i]
        _t.append(t)
        temp = factor.ix[t].dropna()
        temp.sort_values(ascending=ascending, inplace=True)
        stock_to_hold = temp[:stock_number].index
        stocks[t] = stock_to_hold
        factors[t] = temp[stock_to_hold]
        _return.append(returns.ix[t][stock_to_hold].mean())
        _mean.append(temp[stock_to_hold].mean())
    _mean = pd.Series(_mean, index=_t, name='factor_mean')
    _return = pd.Series(_return, index=_t, name='mean_return')
    return {'factor_mean': _mean, 'mean_return': _return,
            'stocks': stocks.T, 'factors': factors.T}


high = import_data('LZ_GPA_QUOTE_THIGH')
low = import_data('LZ_GPA_QUOTE_TLOW')
close = import_data('LZ_GPA_QUOTE_TCLOSE')
OPEN = import_data('LZ_GPA_QUOTE_TOPEN')
value = import_data('LZ_GPA_QUOTE_TVALUE')
volume = import_data('LZ_GPA_QUOTE_TVOLUME')
stop = import_data('LZ_GPA_SLCIND_STOP_FLAG')
price_adj_f = import_data('LZ_GPA_CMFTR_CUM_FACTOR')

data = {'high': high,
        'low': low,
        'close': close,
        'OPEN': OPEN,
        'value': value,
        'volume': volume,
        'stop': stop,
        'price_adj_f': price_adj_f}


factor_001 = factor_cal(data, '001', no_zdt=False)
factor_006 = factor_cal(data, '006', no_zdt=False)
factors = pd.concat([factor_003.stack(), factor_006.stack()], axis=1)
days = factors.index.levels[0]
for d in days:
    print factors.ix[d].corr().iloc[0][1]


factor_ = factor_cal()
factor_.to_csv('F:\Factors\Volatility/LowVol_nostop_adj.csv')
factor_ = factor_.rolling(window=12).mean()
temp_ = factor_.stack()
temp_.plot(kind='hist', bins=20)
temp_[temp_>0.1]

summary = factor_summary(factor_['2013-01-01':].dropna(axis=0, how='all'))
summary.ix[:, ['mean', 'std', '10%', '90%']].plot(figsize=(10, 10))

ic_series = ic(factor_, 'close-close', 20, data, by_day=True)
title = '60 days moving average of IC (absolute value)'
np.abs(ic_series).rolling(window=60).mean().plot(figsize=(14, 7), title=title)


temp = np.abs(factor_.T - factor_.mean(axis=1)).T
factor_name = 'alpha#101_nozdt_abs_1D'
temp.index.name = factor_name
temp.to_csv('F:\Strategies\World_Quant_Alphas\#%s\%s.csv' %
            ('101', factor_name))

stt = factor_['2013-01-07':].dropna(axis=0, how='all')
stt = np.abs(stt.T - stt.mean(axis=1)).T
stt.mean(axis=1).mean()

a = group_analysis(stt, data, 'close-close', period=20, bins=10, cut_mode='quantile')
a['group_return'].mean().plot(kind='bar')
a['group_mean'].mean().plot()
(a['group_return'] + 1).cumprod().plot(figsize=(14, 7))
a['group_mean']['Factor Average'].plot(figsize=(14, 7))
a['group_return'].columns.name

a = _analysis(stt, data, 'close-close', stock_number=50, ascending=True, period=10)
a['mean_return'].mean()
a['factor_mean'].min()
(a['mean_return'] + 1).cumprod().plot(figsize=(14, 7))
a['stocks'].to_csv('F:\Strategies\World_Quant_Alphas/10_stocks.csv')
a['factors'].head()



